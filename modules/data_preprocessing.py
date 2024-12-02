import pandas as pd
from datasets import Dataset
from ast import literal_eval

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
            'explanation': row.get('explanation', None),
        }
        records.append(record)

    df = pd.DataFrame(records)
    df['question_plus'] = df['question_plus'].fillna('')
    return Dataset.from_pandas(df)

def format_dataset(dataset, prompt_args):
    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])
        cot_example = dataset[i].get("explanation", "이 문제를 해결하기 위해 지문에서 필요한 정보를 추출하고, 질문과 연관지어 정답을 유추합니다.")

        if dataset[i]["question_plus"]:
            user_message = prompt_args.PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = prompt_args.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )

        assistant_message = f"{cot_example}\n정답: {dataset[i]['answer']}"

        if prompt_args.use_cot:
            processed_dataset.append({
            "id": dataset[i]["id"],
            "messages": [
                {"role": "system", "content": prompt_args.system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ],
            "label": assistant_message,
            })
        else:
            processed_dataset.append({
                "id": dataset[i]["id"],
                "messages": [
                    {"role": "system", "content": prompt_args.system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                ],
                "label": dataset[i]["answer"],
            })

    return Dataset.from_pandas(pd.DataFrame(processed_dataset))

def format_inference_dataset(test_df, prompt_args):
    test_dataset = []
    for i, row in test_df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])
        
        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = prompt_args.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = prompt_args.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": prompt_args.system_message},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )

    return test_dataset