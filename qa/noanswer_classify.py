#!/usr/bin/env python3
"""Use a finetuned noanswer classification model trained with `noanswer-train.py`
and apply it on a dataset.
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

INPUT_FIELD = "text"

def noanswer_classify(
        model_path: str,
        question_file: str,
        context_file: str,
        output_path: str,
        batch_size: int = 32):


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model_path,
                          tokenizer=tokenizer, device=0, truncation=True)


    questions_f = open(question_file, "r", encoding="utf-8")
    contexts_f = open(context_file, "r", encoding="utf-8")

    data = []
    for question, context in zip(questions_f, contexts_f):
        question = question.strip()
        context = context.strip()

        data.append(f"{question} {context}")

    questions_f.close()
    contexts_f.close()

    results = classifier(data, batch_size=batch_size)

    with open(output_path, "w") as f:
        for result in results:
            f.write(f"{result['label']}\n")


if __name__ == "__main__":
    print("Debug mode")
    noanswer_classify(model_path="all-translated/checkpoint-3795",
                      question_file="test-all.question.eng",
                      context_file="test-all.context.eng",
                      output_path="debug-results.eng")
