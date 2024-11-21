from datasets import Dataset

def tokenize_dataset(dataset, tokenizer):
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts
    
    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    return dataset.map(
        tokenize,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

def split_dataset(tokenized_dataset):
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)  
    return tokenized_dataset.train_test_split(test_size=0.1, seed=42)
