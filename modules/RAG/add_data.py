import pandas as pd
import json

augmented_file = "llm_keyword_ret.csv"
wiki_file = "wiki.jsonl"
output_file = "final_train.csv"

def get_wiki_content_by_title(title, wiki_file):
    with open(wiki_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            if doc['title'] == title:
                return doc['content']
    return None

def extract_first_sentences(content, num_sentences=2):
    if not content:
        return ""
    sentences = content.split('. ')
    return '. '.join(sentences[:num_sentences]).strip() + ('.' if sentences else '')

augmented_df = pd.read_csv(augmented_file)

updated_data = []

for i, row in augmented_df.iterrows():
    paragraph = row['paragraph']
    retrieve_terms = eval(row['retrieve'])  
    
    additional_info = []
    for term in retrieve_terms:
        content = get_wiki_content_by_title(term, wiki_file)
        if content:
            additional_info.append(extract_first_sentences(content))
    
    combined_paragraph = paragraph + " " + " ".join(additional_info)
    row['updated_paragraph'] = combined_paragraph
    updated_data.append(row)

updated_df = pd.DataFrame(updated_data)
updated_df.to_csv(output_file, index=False, encoding="utf-8-sig")