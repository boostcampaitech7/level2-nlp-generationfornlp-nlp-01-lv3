from modules.data_preprocessing import load_and_process_data, format_dataset
from modules.model_utils import load_model_and_tokenizer, get_peft_config
from modules.training_utils import tokenize_dataset, split_dataset
from modules.set_seed import set_seed
from modules.evaluation import preprocess_logits_for_metrics, compute_metrics
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from omegaconf import OmegaConf
import argparse


def main(config):
    set_seed(42)
    model_args, data_args, train_args = config.model, config.data, config.train
    dataset = load_and_process_data(data_args.train_csv)
    formatted_dataset = format_dataset(dataset)

    model, tokenizer = load_model_and_tokenizer(model_args.model_name_or_path)
    peft_config = get_peft_config()

    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer)
    train_dataset, eval_dataset = split_dataset(tokenized_dataset).values()

    response_template = "<start_of_turn>model"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(**train_args)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_res: compute_metrics(eval_res, tokenizer),
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, tokenizer),
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config_path", default="config/config.yaml", type=str, help="Path to config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    main(config)
