import argparse
import os
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from modules.data_preprocessing import load_and_process_data, format_inference_dataset
from modules.model_utils import load_inference_model_and_tokenizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, config):
    setup(rank, world_size)
    model_args, data_args, prompt_args = config.model, config.data, config.prompt

    # 모델과 토크나이저 로드
    model, tokenizer = load_inference_model_and_tokenizer(model_args.model_name_or_path, model_args.load_in_8b)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    # 데이터 로드 및 전처리
    test_df = load_and_process_data(data_args.inference_csv).to_pandas()
    test_dataset = format_inference_dataset(test_df, prompt_args)  # test_dataset은 list of dict

    infer_results = []

    with torch.inference_mode():
        for data in tqdm(test_dataset, total=len(test_dataset), disable=rank != 0):
            input_tensor = tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(rank)

            outputs = model.module.generate(
                input_tensor,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

            response = tokenizer.decode(outputs[0][input_tensor.size(1):], skip_special_tokens=True)

            infer_results.append({
                "id": data["id"],
                "answer": response.strip()
            })

    # 결과 저장 (주 프로세스에서만 수행)
    if rank == 0:
        pd.DataFrame(infer_results).to_csv(data_args.output_csv, index=False)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config_path", default="config/config.yaml", type=str, help="Path to config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config), nprocs=world_size)
