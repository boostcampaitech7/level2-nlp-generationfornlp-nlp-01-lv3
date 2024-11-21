from modules.data_preprocessing import load_and_process_data, format_dataset
from modules.model_utils import load_model_and_tokenizer
from modules.training_utils import tokenize_dataset, split_dataset
from modules.set_seed import set_seed
from modules.evaluation import preprocess_logits_for_metrics, compute_metrics
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from omegaconf import OmegaConf
import argparse
from peft import LoraConfig


def main(config):
    set_seed(config.seed)
    model_args, data_args, train_args, prompt_args, peft_args = config.model, config.data, config.train, config.prompt, config.peft
    dataset = load_and_process_data(data_args.train_csv)
    formatted_dataset = format_dataset(dataset, prompt_args)

    model, tokenizer = load_model_and_tokenizer(model_args.model_name_or_path, prompt_args.tokenizer_chat_template)
    peft_config = LoraConfig(**peft_args)

    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer)
    train_dataset, eval_dataset = split_dataset(tokenized_dataset, data_args, config.seed).values()

    response_template = prompt_args.response_template
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
        compute_metrics=lambda eval_res: compute_metrics(eval_res, tokenizer, prompt_args.compute_metrics_end_token),
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
