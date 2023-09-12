#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, DefaultDataCollator, \
    AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset


def finetune_qa(
        train_tsv,
        valid_tsv,
        output_dir,
        model_type="deepset/roberta-large-squad2",
        gradient_accumulation_steps=4,
        epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.0):

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
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=learning_rate,
        auto_find_batch_size=True,
        num_train_epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator)

    trainer.train()

    best_checkpoint = trainer.state.best_model_checkpoint
    print("Best checkpoint: {}".format(best_checkpoint))

    os.symlink(best_checkpoint, os.path.join(output_dir, "best_checkpoint"))


if __name__ == "__main__":
    print("Debug mode")
    finetune_qa(
        "qa.tiny.train.tsv",
        "qa.tiny.valid.tsv",
        "qa.foo_dir")
