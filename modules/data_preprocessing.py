import pandas as pd
from datasets import Dataset
from ast import literal_eval

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

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
        }
        records.append(record)

    df = pd.DataFrame(records)
    return Dataset.from_pandas(df)

def format_dataset(dataset):
    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

        if dataset[i]["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )

        processed_dataset.append({
            "id": dataset[i]["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{dataset[i]['answer']}"}
            ],
            "label": dataset[i]["answer"],
        })

    return Dataset.from_pandas(pd.DataFrame(processed_dataset))