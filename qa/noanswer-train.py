#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np

INPUT_FIELD = "text"
LABEL_FIELD = "label"


def train_noanswer_classification(
        train_tsv,
        valid_tsv,
        output_dir,
        model_type="deepset/roberta-large-squad2",
        batch_size=8,
        epochs=3,
        learning_rate=2e-5):

    dataset = load_dataset(
        'csv',
        data_files={'train': train_tsv, 'validation': valid_tsv},
        delimiter='\t')

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    def tokenize(batch):
        return tokenizer(batch[INPUT_FIELD], truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        #save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)

    trainer.train()


if __name__ == "__main__":
    print("Debug mode")
    train_noanswer_classification(
        train_tsv="valid-all.eng.tsv",
        valid_tsv="test-all.sample.eng.tsv",
        output_dir="all-translated",
        model_type="deepset/roberta-large-squad2")
