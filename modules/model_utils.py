from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch

def load_model_and_tokenizer(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    return model, tokenizer

def get_peft_config():
    return LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj'],
        bias="none",
        task_type="CAUSAL_LM",
    )
