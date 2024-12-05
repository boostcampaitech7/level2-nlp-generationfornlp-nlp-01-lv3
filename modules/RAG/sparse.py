from rank_bm25 import BM25Okapi
import pandas as pd
import json

new_train_file = "new_train.csv"  
wiki_file = "wiki.jsonl"    

new_train = pd.read_csv(new_train_file)

wiki_titles = []
wiki_contents = []

with open(wiki_file, 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line)
        wiki_titles.append(doc['title'])
        wiki_contents.append(doc['content'])

tokenized_titles = [title.split() for title in wiki_titles]
bm25 = BM25Okapi(tokenized_titles)

def sparse_retrieve_with_bm25(question, choices, top_k=5):
    query = f"{question} {' '.join(choices)}"
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    retrieved_titles = [wiki_titles[i] for i in top_indices]
    return retrieved_titles

augmented_data = []
for i, row in new_train.iterrows():
    question = row['question']
    choices = eval(row['choices']) 

    retrieve_titles = sparse_retrieve_with_bm25(question, choices, top_k=5)

    row['retrieve'] = retrieve_titles
    augmented_data.append(row)

    if i % 100 == 0:
        print(f"Processed {i}/{len(new_train)} rows")

augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_csv("sparse_ret.csv", index=False, encoding="utf-8-sig")