import logging

def activate_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO)

MASAKHANER_LANGUAGES = [
    "amh", "hau", "ibo", "kin", "kin", "luo", "pcm", "swa", "wol", "yor"]

NLLB_CODES = {
    "amh": "amh_Ethi",
    "hau": "hau_Latn",
    'ibo': "ibo_Latn",
    'kin': "kin_Latn",
    'luo': "luo_Latn",
    'pcm': "pcm_Latn",
    'swa': "swh_Latn",
    'wol': "wol_Latn",
    'yor': "yor_Latn",
}

TAG_LIST = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE",
    "I-DATE"]

SPACY_TYPES = ["PERSON", "ORG", "LOC", "DATE", "GPE", "TIME"]


rule all:
    input:
        #expand("masakhaner_data/{lang}/{split}.eng",
        #       lang=MASAKHANER_LANGUAGES, split=["dev", "test", "train"]),
        #expand("masakhaner_experiments/spacy/{lang}/{split}-retok.eng",
        #       lang=MASAKHANER_LANGUAGES, split=["dev", "test"]),
        "masakhaner_experiments/spacy/amh/test.score",


rule download_masakhaner:
    output:
        text_file="masakhaner_data/{lang}/{split}.txt",
        tags_file="masakhaner_data/{lang}/{split}.tags"
    run:
        import datasets
        test_data = datasets.load_dataset("masakhaner", wildcards.lang)['test']
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
        flags="--gres=gpu:1 --constraint='gpuram48G|gpuram40G'"
    run:
        from constrained_translate_with_tags import project_markup
        tags = project_markup(
            open(input.retokenized).read().splitlines(),
            open(input.tags).read().splitlines(),
            open(input.target_text).read().splitlines(),
            language=NLLB_CODES[wildcards.lang],
            model="ychenNLP/nllb-200-3.3B-easyproject",
            batch_size=16,
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

        with open(output.report, "w") as f:
            print(json.dumps(score, indent=4), file=f)
        with open(output.score, "w") as f:
            print(score['overall_f1'], file=f)

        f_output.close()
        f_gold.close()















