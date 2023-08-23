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
            start_sentence_id = 0
            while sent_offsets[start_sentence_id + 1] <= answer_start:
                start_sentence_id += 1
            end_sentence_id = start_sentence_id
            while sent_offsets[end_sentence_id + 1] <= answer_end:
                end_sentence_id += 1
            start_sentence_start = sent_offsets[start_sentence_id]
            end_sentence_start = sent_offsets[end_sentence_id]
            end_sentence_end = sent_offsets[end_sentence_id + 1]
            in_sent_answer_start = answer_start - start_sentence_start
            in_sent_answer_end = answer_end - end_sentence_start

            print(context[start_sentence_start:end_sentence_end])
            print(context[answer_start:answer_end])
            assert context[answer_start:answer_end] in context[start_sentence_start:end_sentence_end]
            print(f"{start_sentence_id}\t{in_sent_answer_start}\t{end_sentence_id + 1}\t{in_sent_answer_end}", file=f_out)
    logger.info("Done.")
