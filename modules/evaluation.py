import evaluate
import numpy as np
import torch

def preprocess_logits_for_metrics(logits, labels, tokenizer):
    """
    모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]

    # Define indices for answer tokens (e.g., "1", "2", ..., "5")
    answer_tokens = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 6)]

    # Extract logits for these answer tokens
    logits = logits[:, -1, answer_tokens]  # Use the last token for answers
    return logits


def compute_metrics(evaluation_result, tokenizer):
    """
    Computes accuracy based on the predicted and true answers.
    """
    acc_metric = evaluate.load("accuracy")
    logits, labels = evaluation_result

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)  # Get predicted answer indices

    # Map predictions to answer tokens
    pred_answers = [str(i + 1) for i in predictions]

    # Decode ground-truth labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Extract only the "정답" portion from the labels
    true_answers = [
        label.split("정답:")[-1].strip().split()[0] for label in decoded_labels
    ]

    # Calculate accuracy
    accuracy = acc_metric.compute(predictions=pred_answers, references=true_answers)

    return {
        "accuracy": accuracy["accuracy"],
    }