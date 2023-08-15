from typing import List

import argparse
import logging

from tner import TransformersNER


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="tner/tner-roberta-large-ontonotes5")
    parser.add_argument("data", type=argparse.FileType("r"))
    args = parser.parse_args()

    logging.info("Loading TNER model.")
    model = TransformersNER(args.model)

    logging.info("Loading English sentences.")
    eng_sentences = args.data.read().splitlines()
    args.data.close()

    logging.info("Applying TNER model.")
    output = model.predict(eng_sentences, batch_size=32)

    logging.info("Writing output.")
    for sentence in output["prediction"]:
        processed_tags = []
        for tag in sentence:
            if tag == "O":
                processed_tags.append(tag)
            if "-" in tag:
                code, ent_type = tag.split("-")
                if ent_type == "PERSON":
                    ent_type = "PER"
                #elif ent_type == "NORP":
                #    ent_type = "ORG"
                elif ent_type == "GPE":
                    ent_type = "LOC"
                elif ent_type == "FAC":
                    ent_type = "LOC"
                elif ent_type == "TIME":
                    ent_type = "DATE"

                if ent_type in ["PER", "ORG", "LOC", "DATE"]:
                    processed_tags.append(f"{code}-{ent_type}")
                else:
                    processed_tags.append("O")
        print(" ".join(processed_tags))

    logging.info("Done.")


if __name__ == "__main__":
    main()
