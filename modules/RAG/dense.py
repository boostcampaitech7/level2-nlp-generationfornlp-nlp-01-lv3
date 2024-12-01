import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import json
import os

new_train_file = "new_train.csv"  
wiki_file = "wiki.jsonl"  

retriever_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

index_file = "faiss_index.bin"
titles_file = "wiki_titles.json"

if os.path.exists(index_file) and os.path.exists(titles_file):
    index = faiss.read_index(index_file)
    with open(titles_file, 'r', encoding='utf-8') as f:
        wiki_titles = json.load(f)
else:
    wiki_documents = []
    wiki_titles = []
    with open(wiki_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            wiki_documents.append(doc['content'])
            wiki_titles.append(doc['title'])

    wiki_embeddings = retriever_model.encode(wiki_documents, show_progress_bar=True)

    dimension = wiki_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(wiki_embeddings)

    faiss.write_index(index, index_file)
    with open(titles_file, 'w', encoding='utf-8') as f:
        json.dump(wiki_titles, f, ensure_ascii=False)

def retrieve_with_rerank(question, choices, top_k=5):
    query = f"{question} {' '.join(choices)}"
    query_embedding = retriever_model.encode([query])

    distances, indices = index.search(query_embedding, top_k)
    retrieved_titles = [wiki_titles[idx] for idx in indices[0]]

    rerank_inputs = [(query, title) for title in retrieved_titles]
    scores = reranker_model.predict(rerank_inputs)
    reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return [retrieved_titles[i] for i in reranked_indices]

new_train = pd.read_csv(new_train_file)

augmented_data = []
for i, row in new_train.iterrows():
    question = row['question']
    choices = eval(row['choices'])  

    retrieve_titles = retrieve_with_rerank(question, choices, top_k=5)

    row['retrieve'] = retrieve_titles
    augmented_data.append(row)

    if i % 100 == 0:
        print(f"Processed {i}/{len(new_train)} rows")

augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_csv("dense_ret.csv", index=False, encoding="utf-8-sig")