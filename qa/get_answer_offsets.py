#!/usr/bin/env python3

NOANSWER = "No Answer"


def get_answer_offsets(context: str, answer: str):
    """context holds a paragraph with senteces separated by tabs, answer is
    contained within context after removing the tabs
    """

    offsets = [0, 0, 0, 0]
    if answer == NOANSWER:
        return offsets


    # remove tabs, do not replace with spaces
    context_notabs = context.replace("\t", "")

    # find the answer in the context
    answer_start = context_notabs.find(answer)
    answer_end = answer_start + len(answer)

    if(answer_start == -1):
        print("ERROR: Answer not found in context")
        print("Answer: " + answer)
        print("Context: " + context_notabs)
        return None

    # find the sentence containing the answer
    current_sentence = 0
    current_offset = 0
    global_offset = 0


    for char in context:
        if char == "\t":
            current_sentence += 1
            current_offset = 0
            continue

        if global_offset == answer_start:
            offsets[0] = current_sentence
            offsets[1] = current_offset

        if global_offset == answer_end:
            offsets[2] = current_sentence + 1
            offsets[3] = current_offset

        current_offset += 1
        global_offset += 1

    return offsets


def reconstruct_answer(context: str, offsets: list[int]):
    if offsets == [0, 0, 0, 0]:
        return NOANSWER

    contex_spl = context.split("\t")

    if offsets[0] == offsets[2] - 1:
        return contex_spl[offsets[0]][offsets[1]:offsets[3]]
    elif offsets[0] < offsets[2] - 1:
        answer = contex_spl[offsets[0]][offsets[1]:]
        for i in range(offsets[0] + 1, offsets[2] - 1):
            answer += contex_spl[i]
        answer += contex_spl[offsets[2] - 1][:offsets[3]]
    else:
        print("ERROR: malformed offsets, cannot end before start")
        return None

    return answer



def test(context, answer):
    offsets = get_answer_offsets(context, answer)
    print(offsets)
    if offsets is not None:
        print(reconstruct_answer(context, offsets))
    print()


if __name__ == "__main__":
    print("Debug mode")

    context = "A dog. \tThe dog's name was Alfons. \tHis favourite toy was a ball."
    print("Context: " + context)
    print()

    print("Test 1. Answer within single sentence. Answer 'Alfons'")
    answer = "Alfons"
    test(context, answer)

    answer = "dog. The dog's name was Alfons. His favourite"
    print("Test 2. Spanning sentence boundaries. Answer '{}'".format(answer))
    test(context, answer)

    print("Test 3. This should print [0,0,0,0] and No answer.")
    answer = "No Answer"
    test(context, answer)

    print("Test 4. This should throw an error.")
    answer = "Hello"
    test(context, answer)
