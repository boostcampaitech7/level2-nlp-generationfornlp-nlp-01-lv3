import evaluate
import numpy as np
import torch

def preprocess_logits_for_metrics(logits, labels, tokenizer):
    """
    모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits

def compute_metrics(evaluation_result, tokenizer, compute_metrics_end_token):
    """
    Compute metrics using the evaluation result (logits and labels).
    """
    acc_metric = evaluate.load("accuracy")
    logits, labels = evaluation_result

    # Map output tokens to integer labels
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    # Replace padding labels with the tokenizer pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split(compute_metrics_end_token)[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    # Apply softmax to logits and get predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # Compute accuracy
    return acc_metric.compute(predictions=predictions, references=labels)
