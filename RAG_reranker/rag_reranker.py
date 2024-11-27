from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import json

#test 데이터
test_file_path = '../rag_data/test.csv'
test_df = pd.read_csv(test_file_path)

# wiki 로드하기
wiki_file_path = '../rag_data/wiki_test.jsonl'
wiki_data = []
with open(wiki_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        wiki_data.append(json.loads(line))

# 질문, text 추출
questions = test_df["problems"].apply(lambda x: eval(x)["question"]).tolist()
documents = [entry["content"] for entry in wiki_data]

# Load tokenizer and model
model_name = "dragonkue/bge-reranker-v2-m3-ko"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute embeddings
def compute_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Compute embeddings for questions and documents
question_embeddings = compute_embeddings(questions)
document_embeddings = compute_embeddings(documents)

# Compute cosine similarity
def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.T)

similarities = cosine_similarity(question_embeddings, document_embeddings)

# Retrieve top documents for each question
top_k = 3  # Number of top documents to retrieve
top_results = []
for question_idx in range(similarities.size(0)):
    similarity_scores = similarities[question_idx]
    top_indices = torch.topk(similarity_scores, top_k).indices
    top_docs = [(wiki_data[idx]["title"], similarity_scores[idx].item()) for idx in top_indices]
    top_results.append({
        "question": questions[question_idx],
        "top_documents": top_docs
    })

# Convert the results to a DataFrame for better visualization
results_df = pd.DataFrame(top_results)

print(results_df.head())
