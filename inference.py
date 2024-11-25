from omegaconf import OmegaConf
import argparse
from modules.data_preprocessing import load_and_process_data, format_inference_dataset
from modules.model_utils import load_inference_model_and_tokenizer
import torch
from tqdm import tqdm
import pandas as pd

def main(config):
    model_args, data_args, prompt_args = config.model, config.data, config.prompt

    model, tokenizer = load_inference_model_and_tokenizer(model_args.model_name_or_path, model_args.load_in_8b)
    model.eval()
    model.to("cuda")

    test_df = load_and_process_data(data_args.inference_csv).to_pandas()
    test_dataset = format_inference_dataset(test_df, prompt_args)  # test_dataset은 list of dict

    infer_results = []

    with torch.inference_mode():
        for data in tqdm(test_dataset, total=len(test_dataset)):
            input_tensor = tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")

            outputs = model.generate(
                input_tensor,
                max_new_tokens=1,
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

    pd.DataFrame(infer_results).to_csv(data_args.output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config_path", default="config/config.yaml", type=str, help="Path to config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
