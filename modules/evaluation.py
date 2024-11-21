import evaluate
import numpy as np
import torch

def compute_metrics(evaluation_result, tokenizer):
    acc_metric = evaluate.load("accuracy")
    logits, labels = evaluation_result

    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    return acc_metric.compute(predictions=predictions, references=labels)
