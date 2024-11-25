from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM
import torch

def load_model_and_tokenizer(model_name_or_path, tokenizer_chat_template):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=True,
        # torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer_chat_template != "default":
        tokenizer.chat_template = tokenizer_chat_template
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    return model, tokenizer

def load_inference_model_and_tokenizer(model_name_or_path, load_in_8b):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
        # load_in_4bit=True,
        # load_in_8bit=load_in_8b,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
