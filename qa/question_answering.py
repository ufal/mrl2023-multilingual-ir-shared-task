from typing import List

from transformers import pipeline


def inference(
        model_name: str,
        questions: List[str],
        contexts: List[str]) -> List[str]:
    """Inference with a question answering model.

    The answers are marked in the contexts using square brackters, so it can
    get later translated. Original square brackets are replaced with normal
    brackets.

    Args:
    model_name: Name of the model to use.
    questions: List of questions.
    contexts: List of contexts.

    Returns:
    Annotated contexts.
    """

    # Load the model.
    qa = pipeline("question-answering", model=model_name)

    # Format data for the model.
    inputs = [
        {"question": question, "context": context}
        for question, context in zip(questions, contexts)]

    # Run inference.
    model_results = qa(inputs)

    # Format the results.
    results = []
    for context, model_result in zip(contexts, model_results):
        start = model_result["start"]
        end = model_result["end"]
        context = context.replace("[", "(").replace("]", ")")
        results.append(
            context[:start] + "[" + context[start:end] + "]" + context[end:])

    return results
