import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


new_train_file = "new_train.csv"  
wiki_file = "wiki.jsonl"  
redirect_file = "redirects.json"

model_id = 'MLP-KTLim/llama3-Bllossom'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

PROMPT = '''당신은 주어진 질문과 부가 정보를 분석하여 가장 중요한 단어를 추출하는 전문가입니다.
아래의 형식을 준수하여, 질문과 부가 정보를 기반으로 중요한 단어를 최대한 많이 추출하고, 중요도 순으로 정렬하세요.

출력 형식:
- 중요한 단어: ["단어1", "단어2", "단어3", ...]

추출 기준:
1. 질문(Question)과 부가 정보(Question_plus)를 이해하는 데 필수적인 고유명사(예: 인물, 장소, 사건 등).
2. 질문과 부가 정보에서 반복적으로 언급되거나 문맥상 중요한 단어.
3. 문장 간의 연결을 이해하는 데 필수적인 동사나 명사(예: 주요 개념, 동작 등).
4. 단어는 반드시 원형으로 추출하며, 중복되지 않게 반환하세요.

아래는 예시입니다.

---

예시 1:
질문: "<보기>를 바탕으로 윗글을 감상한 내용으로 적절하지 않은 것은?"
선택지: ["옥영의 꿈에 나타난 ‘만복사의 부처’ 는...", "몽석의 몸에 나타난 ‘붉은 점’ 은..."]
부가 정보: "｢최척전｣에는 하나의 문제 상황이 해결되면 또 다른 문제가 확인되는 서사 구조가 나타나고 있다."
- 중요한 단어: ["최척전", "만복사", "몽석", "붉은 점", "서사 구조"]

---

입력된 질문과 선택지와 부가 정보를 바탕으로 위와 같은 기준으로 중요한 단어를 추출하고, 반드시 위의 형식으로 반환하세요.
'''

def load_wiki_titles(wiki_file):
    wiki_titles = set()
    with open(wiki_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            wiki_titles.add(doc['title'])
    return wiki_titles

def extract_keywords_with_llm(question, choices):
    query = f"질문: {question}\n선택지: {choices}"
    input_ids = tokenizer(query, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids['input_ids'],
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        keywords = json.loads(result)
    except (json.JSONDecodeError, IndexError):
        keywords = []
    return keywords

def filter_terms(term_list, wiki_titles):
    matched_terms = []
    for term in term_list:
        if term in wiki_titles:
            matched_terms.append(term)
        else:
            no_space_term = term.replace(" ", "")
            if no_space_term in wiki_titles:
                matched_terms.append(no_space_term)
    return matched_terms

def apply_redirect(terms, redirect_map):
    return [redirect_map.get(term, term) for term in terms]

def remove_excluded_terms(terms, exclude_terms):
    return [term for term in terms if term not in exclude_terms]

new_train = pd.read_csv(new_train_file)
wiki_titles = load_wiki_titles(wiki_file)
redirect_data = json.load(open(redirect_file, 'r', encoding='utf-8'))
redirect_map = {item['source_title']: item['redirect_to'] for item in redirect_data}
exclude_terms = {'갑', '을', '임', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'A', 'B', 'C', 'D', 'E', '가', '나', '사상가'}

augmented_data = []

for i, row in new_train.iterrows():
    question = row['question']
    choices = eval(row['choices'])
    keywords = extract_keywords_with_llm(question, choices)
    filtered_keywords = filter_terms(keywords, wiki_titles)
    filtered_keywords = remove_excluded_terms(filtered_keywords, exclude_terms)
    final_keywords = apply_redirect(filtered_keywords, redirect_map)
    row['keywords'] = final_keywords
    augmented_data.append(row)

augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_csv("llm_keyword_ret.csv", index=False, encoding="utf-8-sig")