from omegaconf import OmegaConf
import argparse
from modules.data_preprocessing import load_and_process_data, format_inference_dataset
from modules.model_utils import load_inference_model_and_tokenizer
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd



def main(config):
    model_args, data_args = config.model, config.data

    model, tokenizer = load_inference_model_and_tokenizer(model_args.model_name_or_path)

    test_df = load_and_process_data(data_args.inference_csv).to_pandas()
    test_dataset = format_inference_dataset(test_df) # test_dataset은 list of dict

    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()
            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(target_logit_list, dtype=torch.float32)
                )
                .detach()
                .cpu()
                .numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})
    
    pd.DataFrame(infer_results).to_csv("outputs/output.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config_path", default="config/config.yaml", type=str, help="Path to config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
