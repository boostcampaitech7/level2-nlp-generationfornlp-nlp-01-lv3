from omegaconf import OmegaConf
import argparse
from modules.data_preprocessing import load_and_process_data, format_inference_dataset
from modules.model_utils import load_inference_model_and_tokenizer
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


def batch_inference(model, tokenizer, batch_data, pred_choices_map):
    """
    한 배치의 데이터를 모델에 전달하고 결과를 반환합니다.
    """
    batch_ids = [data["id"] for data in batch_data]
    batch_messages = [data["messages"] for data in batch_data]
    batch_len_choices = [data["len_choices"] for data in batch_data]

    # 입력 텐서를 생성합니다.
    tokenized_inputs = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # 모델 추론
    outputs = model(tokenized_inputs)
    logits = outputs.logits[:, -1].detach().cpu()

    # 결과 계산
    infer_results = []
    for i, logits_per_sample in enumerate(logits):
        len_choices = batch_len_choices[i]
        target_logit_list = [logits_per_sample[tokenizer.vocab[str(j + 1)]].item() for j in range(len_choices)]

        probs = torch.nn.functional.softmax(
            torch.tensor(target_logit_list, dtype=torch.float32),
            dim=-1
        ).numpy()

        predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
        infer_results.append({"id": batch_ids[i], "answer": predict_value})
    
    return infer_results


def main(config):
    model_args, data_args = config.model, config.data

    model, tokenizer = load_inference_model_and_tokenizer(model_args.model_name_or_path)
    test_df = load_and_process_data(data_args.inference_csv).to_pandas()
    test_dataset = format_inference_dataset(test_df)  # test_dataset은 list of dict

    # 배치 관련 설정
    batch_size = config.data.batch_size
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    infer_results = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch_data = test_dataset[i : i + batch_size]  # 배치 단위로 데이터 분할
            batch_results = batch_inference(model, tokenizer, batch_data, pred_choices_map)
            infer_results.extend(batch_results)

    pd.DataFrame(infer_results).to_csv("outputs/output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config_path", default="config/config.yaml", type=str, help="Path to config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
