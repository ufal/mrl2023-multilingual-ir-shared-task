import logging

def activate_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO)

MASAKHANER_LANGUAGES = [
    "amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]

QA_LANGUAGES = [
    "ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te",
]

NLLB_CODES = {
    "amh": "amh_Ethi",
    "ar": "ara_Arab",
    "bn": "ben_Beng",
    "en": "eng_Latn",
    "fi": "fin_Latn",
    "hau": "hau_Latn",
    'ibo': "ibo_Latn",
    "id": "ind_Latn",
    'kin': "kin_Latn",
    'lug': "lug_Latn",
    'luo': "luo_Latn",
    'ko': "kor_Hang",
    'pcm': "eng_Latn",
    'ru': "rus_Cyrl",
    'spa': "spa_Latn",
    'swa': "swh_Latn",
    'sw': "swh_Latn",
    'tha': "tha_Thai",
    'te': "tel_Telu",
    'vie': "vie_Latn",
    'wol': "wol_Latn",
    'yor': "yor_Latn",
    'zho': "zho_Hans"
}


ENTS = ["PER", "ORG", "LOC", "DATE"]
TAG_LIST = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE",
    "I-DATE"]

SPACY_TYPES = ["PERSON", "ORG", "LOC", "DATE", "GPE", "TIME"]


rule all:
    input:
        #expand("masakhaner_data/{lang}/{split}.eng",
        #       lang=MASAKHANER_LANGUAGES, split=["validation", "test", "train"]),
        #expand("masakhaner_experiments/spacy/{lang}/{split}.score",
        #       lang=MASAKHANER_LANGUAGES, split=["validation", "test"]),
        expand("qa_data/{lang}/{type}.{file}.eng",
               lang=QA_LANGUAGES,
               type=["train", "validation", "test"],
               file=["context", "question"]),
        expand("qa_experiments/baseline/{lang}/{type}.eng-annotated",
               lang=QA_LANGUAGES,
               type=["validation", "test"]),


rule download_masakhaner:
    output:
        text_file="masakhaner_data/{lang}/{split}.txt",
        tags_file="masakhaner_data/{lang}/{split}.tags"
    run:
        import datasets
        test_data = datasets.load_dataset(
            "masakhaner", wildcards.lang)[wildcards.split]
        f_text = open(output.text_file, "w")
        f_tags = open(output.tags_file, "w")
        for item in test_data:
            f_text.write(
                " ".join(item['tokens']) + "\n")
            f_tags.write(
                " ".join(TAG_LIST[idx] for idx in item['ner_tags']) + "\n")

        f_text.close()
        f_tags.close()


rule translate_input_text:
    input:
        src_text="{dataset}/{lang}/{split}.txt"
    output:
        tgt_text="{dataset}/{lang}/{split}.eng"
    resources:
        mem="40G",
        cpus_per_task=8,
        partition="gpu-troja,gpu-ms",
        flags="--gres=gpu:1"
    run:
        from transformers import pipeline
        import torch

        logger = logging.getLogger("translate")
        logger.setLevel(logging.INFO)

        logger.info("Loading model.")
        pipe = pipeline(
            "translation", model="facebook/nllb-200-3.3B", device=0)
        logger.info("Model loaded. Loading source text.")
        with open(input.src_text) as f:
            src_text = f.readlines()
        logger.info("Source text loaded. Translating.")
        translation = pipe(
            src_text,
            src_lang=NLLB_CODES[wildcards.lang],
            tgt_lang="eng_Latn")
        logger.info("Translation done. Writing to file.")
        with open(output.tgt_text, "w") as f:
            for line in translation:
                print(line['translation_text'], file=f)
        logger.info("Done.")


rule ner_with_spacy:
    input:
        "masakhaner_data/{lang}/{split}.eng"
    output:
        retokenized="masakhaner_experiments/spacy/{lang}/{split}-retok.eng",
        tags="masakhaner_experiments/spacy/{lang}/{split}.eng-tags",
    resources:
        mem="10G",
        cpus_per_task=4,
    run:
        import spacy
        logger = logging.getLogger("spacy")
        logger.setLevel(logging.INFO)

        logger.info("Loading model.")
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.info("Model not found. Downloading.")
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")

        logger.info("Model loaded. Processing input.")
        f_retokenized = open(output.retokenized, "w")
        f_tags = open(output.tags, "w")
        for line in open(input[0]):
            doc = nlp(line.strip())
            tokens = []
            tags = []
            for token in doc:
                tokens.append(token.text)
                if token.ent_iob_ == "O" or token.ent_type_ not in SPACY_TYPES:
                    tags.append("O")
                else:
                    ent_type = token.ent_type_
                    if ent_type == "PERSON":
                        ent_type = "PER"
                    elif ent_type == "GPE":
                        ent_type = "LOC"
                    elif ent_type == "TIME":
                        ent_type = "DATE"
                    tags.append(token.ent_iob_ + "-" + ent_type)
            print(" ".join(tokens), file=f_retokenized)
            print(" ".join(tags), file=f_tags)
        f_retokenized.close()
        f_tags.close()
        logger.info("Done.")


rule project_from_eng:
    input:
        retokenized="{dataset}_experiments/{experiment}/{lang}/{split}-retok.eng",
        tags="{dataset}_experiments/{experiment}/{lang}/{split}.eng-tags",
        target_text="{dataset}_data/{lang}/{split}.txt",
    output:
        "{dataset}_experiments/{experiment}/{lang}/{split}.orig-tags"
    resources:
        mem="40G",
        cpus_per_task=8,
        partition="gpu-troja,gpu-ms",
        flags="--gres=gpu:1 --constraint='gpuram48G'"
    run:
        from constrained_translate_with_tags import project_markup
        tags = project_markup(
            open(input.retokenized).read().splitlines(),
            open(input.tags).read().splitlines(),
            open(input.target_text).read().splitlines(),
            language=NLLB_CODES[wildcards.lang],
            model="ychenNLP/nllb-200-3.3B-easyproject",
            batch_size=8,
            max_span_len=5)

        with open(output[0], "w") as f:
            for line in tags:
                print(" ".join(line), file=f)


rule evaluate_ner:
    input:
        system_output="{dataset}_experiments/{experiment}/{lang}/{split}.orig-tags",
        gold_standard="{dataset}_data/{lang}/{split}.tags",
    output:
        score="{dataset}_experiments/{experiment}/{lang}/{split}.score",
        report="{dataset}_experiments/{experiment}/{lang}/{split}.report"
    run:
        import evaluate
        import json
        metric = evaluate.load("seqeval")

        with open(input.system_output) as f:
            outputs = [line.strip().split() for line in f]
        with open(input.gold_standard) as f:
            golds = [line.strip().split() for line in f]

        score = metric.compute(predictions=outputs, references=golds)
        for ent in ENTS:
            score[ent]["number"] = int(score[ent]["number"])

        with open(output.report, "w") as f:
            print(json.dumps(score, indent=4), file=f)
        with open(output.score, "w") as f:
            print(score['overall_f1'], file=f)


rule download_xtreme_up:
    output:
        expand("xtreme_up_v1.1/qa_in_lang/{split}/{lang}.jsonl",
               split=['train', 'dev', 'test'],
               lang=QA_LANGUAGES)
    shell:
        """
        wget https://storage.googleapis.com/xtreme-up/xtreme-up-v1.1.jsonl.tgz
        tar -xzf xtreme-up-v1.1.jsonl.tgz
        """


rule format_xtreme_up_qa:
    input:
        "xtreme_up_v1.1/qa_in_lang/{split}/{lang}.jsonl"
    output:
        question="qa_data/{lang}/{split}.question.txt",
        context="qa_data/{lang}/{split}.context.txt",
        anwser="qa_data/{lang}/{split}.answers",
    run:
        import json
        with open(input[0]) as f:
            data = [json.loads(line) for line in f]
        with open(output.question, "w") as f_question, \
             open(output.context, "w") as f_context, \
             open(output.anwser, "w") as f_answer:
            for item in data:
                print(item['question'], file=f_question)
                print(f"{item['title']}. {item['context']}", file=f_context)
                print(item['target'], file=f_answer)


rule baseline_qa:
    input:
        question="qa_data/{lang}/{split}.question.txt",
        context="qa_data/{lang}/{split}.context.txt",
    output:
        "qa_experiments/baseline/{lang}/{split}.eng_annotated",
    resources:
        mem="40G",
        cpus_per_task=8,
        partition="gpu-troja,gpu-ms",
        flags="--gres=gpu:1"
    run:
        import question_answering

        results = question_answering.predict(
            model="deepset/deberta-v3-base-squad2"
            questions=open(input.question).read().splitlines(),
            contexts=open(input.context).read().splitlines())
