#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, DefaultDataCollator, \
    AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset


def finetune_qa(
        train_tsv,
        valid_tsv,
        output_dir,
        model_type="deepset/roberta-large-squad2",
        batch_size=16,
        epochs=3,
        learning_rate=2e-5):

    dataset = load_dataset(
        "csv",
        data_files={"train": train_tsv, "validation": valid_tsv},
        quoting=3, # This is to tell pandas to turn off quoting (in case there is odd number of quotes in the text
        delimiter="\t")

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    def preprocess_function(examples):

        answers = examples["answer"]
        contexts = examples["context"]
        questions = [q.strip() for q in examples["question"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length")

        offset_mapping = inputs.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            context = contexts[i]

            start_char = context.find(answer)
            end_char = start_char + len(answer)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_data = dataset.map(preprocess_function, batched=True,
                                 remove_columns=dataset["train"].column_names)

    data_collator = DefaultDataCollator()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForQuestionAnswering.from_pretrained(model_type).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator)

    trainer.train()


if __name__ == "__main__":
    print("Debug mode")
    finetune_qa(
        "qa.tiny.train.tsv",
        "qa.tiny.valid.tsv",
        "qa.foo_dir")
