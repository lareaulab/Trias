from datasets import load_dataset
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    BartForConditionalGeneration,
)
import torch

from tokenizer import TriasTokenizer
from configuration import TriasConfig
from argument_class import DatasetArguments, ModelArguments
from trainer import CustomTrainer
from datacollator import DataCollatorForBART
from utils import *


def main():
    ### Load arguments 
    parser = HfArgumentParser((DatasetArguments, ModelArguments, Seq2SeqTrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    ### Load the configuration
    config = TriasConfig()
    filtered_model_args = {k: v for k, v in vars(model_args).items() if v is not None}
    config.update(filtered_model_args)

    ### Initialize the tokenizer
    tokenizer = TriasTokenizer(vocab=data_args.vocab_file, 
                               model_max_length=data_args.max_seq_len, 
                               separate_vocabs=False,
                               use_fast=True)
    
    ### Load the model from a checkpoint if given, otherwise initialize a new model
    if training_args.resume_from_checkpoint:
        model = BartForConditionalGeneration.from_pretrained(training_args.resume_from_checkpoint, config=config)
    else:
        model = BartForConditionalGeneration(config=config)
    
    ### Load and tokenize data 
    dataset = custom_load_dataset(data_args.dataset_name, streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42)

    def tokenize(element):
        # Create protein strings with species tags
        protein_with_species_tag = [
            f">>{species}<< {protein}"
            for species, protein in zip(element["species_name"], element["protein"])
        ]
        # extract CDS from mRNA
        cds = [
            mrna[start:end]
            for mrna, start, end in zip(element["mrna"], element["codon_start"], element["codon_end"])
        ]
        # toknize
        tokenized_input = tokenizer(protein_with_species_tag, 
                                    text_target=cds, 
                                    truncation=True,
                                    padding=True)
        return tokenized_input

    train_dataset = shuffled_dataset["train"].map(tokenize, batched=True, remove_columns=shuffled_dataset["train"].column_names)
    val_dataset = None
    if "val" in shuffled_dataset:
        val_dataset = shuffled_dataset["val"].map(tokenize, batched=True, remove_columns=shuffled_dataset["val"].column_names)

    data_collator = DataCollatorForBART(tokenizer, model, mlm_probability=0.15)

    ### Train model    
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_len=data_args.train_len,
    )

    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
