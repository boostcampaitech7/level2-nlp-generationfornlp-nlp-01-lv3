import torch
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# bitsandbytes 안 깔려있으면 주석 해제 후 설치
# !pip install bitsandbytes

model_name_or_path = 'Qwen/Qwen2.5-14B-Instruct'
inference_csv = 'data/test.csv'
output_csv = 'outputs/CoT_zeroshot_Qwen14B.csv'

system_message = "You are an intelligent assistant that solves multiple-choice questions.\
Follow the provided reasoning framework and ensure answers are presented in a structured JSON format.
"

PROMPT_NO_QUESTION_PLUS = """아래는 문제 풀이를 위한 지문과 선택지입니다. 주어진 지문을 읽고 질문에 답해주세요. \
문제 풀이 과정과 정답은 Chain of Thought 방식으로 작성하며, Structured output 형식으로 결과를 제시하세요.

지문:
{paragraph}

질문:
{question}

선택지:
{choices}

풀이과정:
1. 문제를 이해합니다.
2. 필요한 정보를 지문과 질문에서 추출합니다.
3. 각 선택지를 논리적으로 평가합니다.
4. 최종적으로 정답을 선택하고 이를 Structured output으로 작성합니다.

출력 형식은 반드시 아래 형식에 따라 작성하세요:

Structured output:
{{
  "reasoning": "풀이 과정 및 논리적 사고의 상세한 서술",
  "answer": "정답 선택지 (예: 1, 2, 3, 4, 5)"
}}"""

PROMPT_QUESTION_PLUS = """아래는 문제 풀이를 위한 지문과 선택지입니다. 주어진 지문을 읽고 질문에 답해주세요. \
문제 풀이 과정과 정답은 Chain of Thought 방식으로 작성하며, Structured output 형식으로 결과를 제시하세요.

지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

풀이과정:
1. 문제를 이해합니다.
2. 필요한 정보를 지문과 질문에서 추출합니다.
3. 각 선택지를 논리적으로 평가합니다.
4. 최종적으로 정답을 선택하고 이를 Structured output으로 작성합니다.

출력 형식은 반드시 아래 형식에 따라 작성하세요:

Structured output:
{{
  "reasoning": "풀이 과정 및 논리적 사고의 상세한 서술",
  "answer": "정답 선택지 (예: 1, 2, 3, 4, 5)"
}}"""

# torch.bfloat16과 load_in_4bit 중 택 1
# bfloat16이 더 빠르나, OOM이 발생할 경우 load_in_4bit 사용
# 추론 시, 긴 지문+보기가 나올 경우 OOM이 발생할 수 있음
# 둘 중, 하나를 주석 처리해주세요.
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, trust_remote_code=True,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_and_process_data(file_path):
    dataset = pd.read_csv(file_path)
    
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
            'explanation': row.get('explanation', None)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df['question_plus'] = df['question_plus'].fillna('')
    return Dataset.from_pandas(df)

def format_inference_dataset(test_df):
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
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )

    return test_dataset

model.eval()
model.to("cuda")

test_df = load_and_process_data(inference_csv).to_pandas()
test_dataset = format_inference_dataset(test_df)  # test_dataset은 list of dict

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
            max_new_tokens=1000,
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

pd.DataFrame(infer_results).to_csv(output_csv, index=False)