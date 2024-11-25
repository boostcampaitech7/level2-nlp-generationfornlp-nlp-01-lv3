from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM
import torch

model_name = "rtzr/ko-gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)
#  "beomi/Llama-3-Open-Ko-8B"  "Bllossom/llama-3.2-Korean-Bllossom-3B" "NCSOFT/Llama-VARCO-8B-Instruct" "nbeerbower/mistral-nemo-wissenschaft-12B"   MoaData/Myrrh_solar_10.7b_3.0
print(model_name)
print(tokenizer.chat_template)
print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer.special_tokens_map)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side = 'right'