#!/usr/bin/env python3

"""Project entity markup from target language (English) to source language.

For each entity in the English, it creates an English sentence with the entity
marked using square brackets, one entitity at a time. On the output side of the
translator, we create a list of candidate markup positions and score the
sentences using a machine-translation system. Then we chose the best scoring
position and say this is the position of the entity in the source language
sentence.

"""

from typing import List, Tuple, IO
import argparse
import logging

import torch
from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, BeamSearchScorer,
        StoppingCriteriaList, MaxLengthCriteria)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO)


def inject_markup(input_str: str, max_len: int) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Inject markup tokens to get possible translation targets.

    Args:
    input_str: Tokenized target sentence.
    max_len: Maximum length of the inserted markup.

    Returns:
    List of positions and list of sentences with markup.
    """
    tokens = input_str.split(" ")
    positions = []
    sentences = []
    for start in range(len(tokens)):
        for end in range(start + 1, start + max_len):
            if end > len(tokens):
                break
            positions.append((start, end))
            sentences.append(
                " ".join(tokens[:start] + ["["] +
                         tokens[start:end] + ["]"] + tokens[end:]))
    return positions, sentences


def read_entities(file: IO[str]) -> List[List[Tuple[str, int, int]]]:
    """Read entities as BIO tags and format as spans.

    Args:
    file: File with entities in BIO format.

    Returns:
    List of lists of entity spans: type, start, end.
    """
    all_entities = []
    for line in file:
        entities = []
        current_entity = None
        current_start = None
        for i, tag in enumerate(line.strip().split(" ")):
            if tag == "O":
                if current_entity is not None:
                    entities.append((current_entity, current_start, i))
                current_entity = None
            if tag.startswith("B-"):
                if current_entity is not None:
                    entities.append((current_entity, current_start, i))
                current_entity = tag[2:]
                current_start = i
        all_entities.append(entities)
    return all_entities


def format_entities(
        sentences: List[str],
        entities: List[Tuple[str, int, int]]) -> List[str]:
    """Format entity spans (type, start, end) as BIO tags.

    Args:
    sentences: List of sentences in target language.
    entities: List of entity spans: type, start, end.

    Returns:
    List of BIO tags.
    """

    for sentence, spans in zip(sentences, entities):
        tokens = sentence.split(" ")
        tags = ["O"] * len(tokens)
        for ent_type, start, end in spans:
            tags[start] = f"B-{ent_type}"
            for i in range(start + 1, end):
                tags[i] = f"I-{ent_type}"
        yield tags


@torch.no_grad()
def project_markup(
        eng_file: List[str],
        ent_file: List[str],
        tgt_file: List[str],
        language: str,
        model: str = "ychenNLP/nllb-200-3.3B-easyproject",
        batch_size: int = 32,
        max_span_len: int = 4,):
    """Project entity markup from target language to source language."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: '%s'", device)
    logging.info("Loading tokenizer '%s'", model)
    tokenizer = AutoTokenizer.from_pretrained(
        #model,
        "facebook/nllb-200-3.3B",
        src_lang="eng_Latn", tgt_lang=language,)

    logging.info("Loading model '%s'", model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device).eval()
    xent = torch.nn.CrossEntropyLoss(reduction="none")
    logging.info("Translating...")

    all_entities = read_entities(ent_file)
    all_projected_entities = []
    for eng_sent, entities, tgt_sent in zip(eng_file, all_entities, tgt_file):

        markup_positions, markup_sentences = inject_markup(tgt_sent, max_span_len)
        src_tokens = eng_sent.split(" ")

        projected_entities = []
        for ent_type, start, end in entities:
            src_sent = " ".join(
                src_tokens[:start] +
                ["["] + src_tokens[start:end] + ["]"] +
                src_tokens[end:])
            logging.info("Source sentence: %s", src_sent)

            scores = []
            # Process in batches to avoid OOM.
            for i in range(0, len(markup_sentences), batch_size):
                mt_tgt = markup_sentences[i:i + batch_size]
                tokenized = tokenizer(
                    [src_sent] * len(mt_tgt), text_target=mt_tgt,
                    return_tensors="pt").to(device)

                logits = model(**tokenized).logits
                bsz, seq_len, vocab_size = logits.shape
                # By definitions, all target sequences are the same length, so we
                # do not need to mask.
                loss_per_token = xent(
                    logits.view(-1, vocab_size),
                    tokenized["labels"].view(-1)).view(bsz, seq_len)
                scores.extend(loss_per_token.sum(dim=1).cpu().numpy().tolist())

            _, (best_start, best_end) = min(zip(scores, markup_positions))
            projected_entities.append((ent_type, best_start, best_end))
        all_projected_entities.append(projected_entities)

    logging.info("Finished translating %d sentences.", len(all_projected_entities))
    logging.info("Formatting entities as BIO tags...")
    all_projected_tags = format_entities(tgt_file, all_projected_entities)

    logging.info("Done.")
    return all_projected_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eng_file", type=argparse.FileType("r"))
    parser.add_argument("ent_file", type=argparse.FileType("r"))
    parser.add_argument("tgt_file", type=argparse.FileType("r"))
    parser.add_argument("language", type=str)
    parser.add_argument("--model", type=str, default="ychenNLP/nllb-200-3.3B-easyproject")
    parser.add_argument("--max_span_len", type=int, default=4)
    args = parser.parse_args()

    projected_tags = project_markup(
        args.eng_file.read().split("\n"),
        args.ent_file.read().split("\n"),
        args.tgt_file.read().split("\n"),
        args.language,
        args.model, args.max_span_len)

    for tags in projected_tags:
        print(" ".join(tags))


if __name__ == "__main__":
    main()
