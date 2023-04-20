import os
import sys

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, get_peft_model_state_dict
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                        DataCollatorForLanguageModeling, Trainer,
                        TrainingArguments, logging, set_seed)

BASE_MODEL = "stabilityai/stablelm-base-alpha-7b"
DATA_PATH = "YOUR-DATA-DIR-HERE"
OUTPUT_DIR = "./stableLM-Tuned"
# training hyperparams
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 3e-4
CUTOFF_LEN = 1024
VAL_SET_SIZE = 2000


# lora hyperparams
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "query_key_value", "xxx"  # workaround for training
]

# llm hyperparams
TRAIN_ON_INPUTS = True 
GROUP_BY_LENGTH = False  
DEVICE_MAP = Accelerator().process_index # or can be Auto with Torch run
WORLD_SIZE =  int(os.environ.get("WORLD_SIZE", 1))
DDP = WORLD_SIZE != 1

gradient_accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE

if DDP:
    DEVICE_MAP = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE


model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_cache=False,  # if you don't have this it will scream about gradient accumulation every step
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=DEVICE_MAP,
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = prepare_model_for_int8_training(
    model,
    output_embedding_layer_name="embed_out",
    layer_norm_names=["layer_norm", "layernorm"])

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM")

model = get_peft_model(model, config)



def generate_prompt(data_point):
    # Can be updated as necessary.  This is a bit less "instruction" tuned than say Alpaca, but you could use that format
    # I've standardized all of my data internally to conform to {"instruction": instruction, "input": "" or input, "ouput": output}
    if data_point["input"]:
        return f"""Human: {data_point["instruction"]} {data_point["input"]}\n\nAssistant: {data_point["output"]}"""
    else:
        return f"""Human: {data_point["instruction"]}\n\nAssistant: {data_point["output"]}"""


def encode(data_point):
    full_prompt = generate_prompt(data_point)
    retval = tokenizer(full_prompt, padding='max_length', truncation=True, max_length=CUTOFF_LEN, return_tensors='pt')
    input_ids = torch.cat((retval.input_ids,torch.tensor([[tokenizer.eos_token_id]],dtype=retval.input_ids.dtype)),1)
    attention_mask = torch.cat((retval.attention_mask,torch.tensor([[1]],dtype=retval.attention_mask.dtype)),1)
    retval = {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask
    }
    return retval

dataset = load_dataset("json", data_files=DATA_PATH) 


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }

if VAL_SET_SIZE > 0:
    train_val = dataset["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
    )
    val_data = (
        train_val["test"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
    )
else:
    train_data = dataset["train"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
    val_data = None


if not DDP and torch.cuda.device_count() > 1:
    model.is_parallelizable=True
    model.model_parallel=True

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=TrainingArguments(
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        ddp_find_unused_parameters=False if DDP else None,
        group_by_length=GROUP_BY_LENGTH,
    ),

    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

old_state_dict = model.state_dict
model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
