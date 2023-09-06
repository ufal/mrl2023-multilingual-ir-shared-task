#!/usr/bin/env python3
"""Use a finetuned noanswer classification model trained with `noanswer-train.py`
and apply it on a dataset.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_FIELD = "text"

def noanswer_classify(
        model_path: str,
        question_file: str,
        context_file: str,
        output_path: str,
        batch_size: int = 32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(batch):
        return tokenizer(batch[INPUT_FIELD], truncation=True)

    # Load dataset from tsv





if __name__ == "__main__":
    noanswer_classify()
