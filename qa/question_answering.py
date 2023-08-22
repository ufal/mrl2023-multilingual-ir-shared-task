from typing import List
import logging

from transformers import pipeline


def question_answering(
        question_file: str,
        context_file: str,
        context_offsets_file: str,
        output_file: str,
        model_name: str = "deepset/roberta-large-squad2") -> List[str]:
    #output:
        #"{dataset}_experiments/baseline/{lang}/{split}.eng-annotated",

    logger = logging.getLogger("qa")
    logger.setLevel(logging.INFO)

    logger.info("Loading data.")
    with open(context_offsets_file) as f_offsets:
        context_offsets = [
            list(map(int, line.strip().split(" ")))
            for line in f_offsets]
    with open(context_file) as f_context:
        contexts = f_context.read().splitlines()
    with open(question_file) as f_question:
        questions = f_question.read().splitlines()

    logger.info("Loading model.")
    qa = pipeline("question-answering", model=model_name)

    logger.info("Model loaded. Formatting inputs.")
    inputs = [
        {"question": question, "context": context}
        for question, context in zip(questions, contexts)]

    logger.info("Running QA inference.")
    results = qa(inputs)

    logger.info("Writing results to file.")
    with open(output_file, "w") as f_out:
        for result, context, sent_offsets in zip(
                results, contexts, context_offsets):
            answer_start = result["start"]
            answer_end = result["end"]
            sentence_id = 0
            while sent_offsets[sentence_id + 1] <= answer_start:
                sentence_id += 1
            sentence_start = sent_offsets[sentence_id]
            sentence_end = sent_offsets[sentence_id + 1]
            answer_start -= sentence_start
            answer_end -= sentence_start

            # Here, we assume the answer do not cross the sentence boundary.
            answer_end = min(
                answer_end, len(context[sentence_start:sentence_end]))
            sentence = context[sentence_start:sentence_end]
            print(f"{context[sentence_start:sentence_end]}\t{sentence_id}\t"
                  f"{answer_start}\t{answer_end}\t", file=f_out)
    logger.info("Done.")
