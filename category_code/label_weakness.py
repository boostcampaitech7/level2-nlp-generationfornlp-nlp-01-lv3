import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from ast import literal_eval
from tqdm import tqdm

checkpoint_path = "./outputs_gemma/checkpoint-2944"
test_df = pd.read_csv('../data/cleaned_category_labeled_train.csv', encoding = 'CP949')

model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
)

# Flatten the JSON dataset
records = []
for _, row in test_df.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        "question_plus": problems.get('question_plus', None),
        "category" : row['category'],
    }
    # Include 'question_plus' if it exists
    if 'question_plus' in problems:
        record['question_plus'] = problems['question_plus']
    records.append(record)
        
# Convert to DataFrame
test_df = pd.DataFrame(records)

PROMPT_NO_QUESTION_PLUS = \
"""지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = \
"""지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

test_dataset = []
for i, row in test_df.iterrows():
    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
    len_choices = len(row["choices"])
    
    # <보기>가 있을 때
    if row["question_plus"]:
        user_message = PROMPT_QUESTION_PLUS.format(
            paragraph=row["paragraph"],
            question=row["question"],
            question_plus=row["question_plus"],
            choices=choices_string,
        )
    # <보기>가 없을 때
    else:
        user_message = PROMPT_NO_QUESTION_PLUS.format(
            paragraph=row["paragraph"],
            question=row["question"],
            choices=choices_string,
        )

    test_dataset.append(
        {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": row["answer"] if row["answer"] != '' else None,
            "len_choices": len_choices,
        }
    )

infer_results = []
pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

model.eval()
with torch.inference_mode():
    for data in tqdm(test_dataset):
        _id = data["id"]
        messages = data["messages"]
        len_choices = data["len_choices"]
        true_answer = data["label"]

        outputs = model(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
        )

        logits = outputs.logits[:, -1].flatten().cpu()

        # 각 선택지에 대한 로그릿을 추출합니다.
        target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

        # 소프트맥스 함수를 적용하여 확률을 계산합니다.
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32)
            )
            .detach()
            .cpu()
            .numpy()
        )

        predicted_index = np.argmax(probs, axis=-1)
        predict_value = pred_choices_map[predicted_index]
        pred_confidence = probs[predicted_index]

        # 상위 두 확률의 차이를 계산하여 모델의 확신 정도를 측정합니다.
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) > 1:
            confidence_gap = sorted_probs[0] - sorted_probs[1]
        else:
            confidence_gap = 0

        # 정답이 있는 경우에만 정확도를 계산합니다.
        if true_answer is not None:
            is_correct = (predict_value == str(true_answer))
        else:
            is_correct = None  # 정답이 없으므로 정확도는 계산하지 않습니다.

        infer_results.append({
            "id": _id,
            "predicted_answer": predict_value,
            "true_answer": str(true_answer) if true_answer is not None else '',
            "is_correct": is_correct,
            "pred_confidence": pred_confidence,
            "confidence_gap": confidence_gap,
            "probs": probs.tolist(),
        })

# 결과를 데이터프레임으로 변환하고 CSV 파일로 저장합니다.
result_df = pd.DataFrame(infer_results)
result_df['category'] = test_df['category']
result_df.to_csv("detailed_output_with_category.csv", index=False)

# category별 취약점 분석
category_analysis = result_df.groupby('category')['is_correct'].mean()
print(category_analysis)