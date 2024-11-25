from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import ast
import csv
from tqdm import tqdm

# 모델과 토크나이저 불러오기
model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# CSV 데이터 불러오기
input_csv_path = 'data/test.csv'  # 입력 CSV 파일 경로
output_csv_path = 'outputs/Qwen-14b-zeroshot.csv'  # 출력 CSV 파일 경로

# 입력 CSV 파일을 pandas DataFrame으로 읽기
data = pd.read_csv(input_csv_path)

def generate_reply(paragraph, problems, question_plus):
    """
    주어진 지문과 문제를 바탕으로 모델을 통해 답변을 생성하는 함수
    """
    try:
        # 'problems' 필드를 딕셔너리로 변환
        problems_dict = ast.literal_eval(problems)
    except Exception as e:
        print(f"problems 필드 파싱 오류: {e}")
        return "problems 필드 파싱 오류"

    # 메시지 포맷팅
    messages = [
        {"role": "system", "content": "지문을 읽고 보기 중에서 올바른 답을 고르는 챗봇 입니다."},
        {"role": "user", "content": f"지문: {paragraph}"},
    ]

    # question_plus가 존재하면 추가
    if pd.notna(question_plus):
        messages.append({"role": "user", "content": f"<보기>: {question_plus}"})
    # 문제와 보기를 추가
    choices_formatted = '\n'.join([f"{i}. {choice}" for i, choice in enumerate(problems_dict.get('choices', []), start=1)])
    messages.append({
        "role": "user",
        "content": f"{problems_dict.get('question', '해당 질문')}.\n{choices_formatted}\n이유를 함께 작성해주세요."
    })

    # 메시지를 텍스트로 변환
    formatted_chat = ""
    for message in messages:
        formatted_chat += f"{message['role']}: {message['content']}\n"
    print(formatted_chat)
    # 토크나이즈
    inputs = tokenizer(
        formatted_chat,
        return_tensors="pt",
        add_special_tokens=False
    )   
    if len(inputs["input_ids"][0]) > 2400:
        return 9999
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 500,  
            pad_token_id=tokenizer.eos_token_id,   
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  
            temperature=None, 
            top_p=None,
            top_k=None
        )
    
    # 생성된 텍스트 디코딩
    response = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    # 새로운 컬럼 'reply' 추가
    data['reply'] = ""

    batch_size = 100  # 배치 크기 설정
    output_csv_path = "output_test_with_replies.csv"  # 덮어쓰기로 저장할 파일

    # 각 행을 순회하며 응답 생성
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Generating replies"):
        paragraph = row['paragraph']
        problems = row['problems']
        question_plus = row.get('question_plus', 'none')

        # 챗봇 응답 생성
        reply = generate_reply(paragraph, problems, question_plus)
        print(reply)

        # 생성된 응답을 DataFrame에 추가
        data.at[index, 'reply'] = reply

        # 100개마다 저장 (덮어쓰기)
        if (index + 1) % batch_size == 0 or (index + 1) == data.shape[0]:  # 마지막 배치도 저장
            data.iloc[:index + 1].to_csv(output_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"{index + 1}개 데이터가 저장되었습니다: {output_csv_path}")

    print(f"모든 응답이 '{output_csv_path}' 파일에 최종 저장되었습니다.")

