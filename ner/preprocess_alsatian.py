#!/usr/bin/env python3

import sys
import re

REPLACE_ALS = {
    "Dr": "Der", "dr": "der", "isch": "ist", "het": "hat", "hett": "hatte",
    "Septämber": "September", "Novämber": "November", "Dezämber": "Dezember",
    "vu": "von", "Vu": "Von", "Vum": "Vom", "vum": "vom", "gsii": "gewesen",
    "gsi": "gewesen", "syt": "seit", "wun": "wenn", "em": "im", "z": "zu",
    "bi": "bei", "gsee": "gesehen", "e": "ein", "as": "als", "äine": "eine",
    "äin": "ein", "uf": "auf", "s": "das", "fir": "für", "sini": "seine",
    "sine": "seine",
}



def main():
    for line in sys.stdin:
        out_words = []
        for word in line.split():
            if word in REPLACE_ALS:
                out_words.append(REPLACE_ALS[word])
                continue
            if word.startswith("g") and len(word) > 2 and word[1] not in "aeiouäöü":
                word = "ge" + word[2:]
            if word.startswith("Joor") or word.startswith("Johr"):
                word = "Jahr" + word[4:]
            if len(word) > 5 and word.endswith("e") and word[0].islower():
                word = word + "n"
            out_words.append(word)
        print(" ".join(out_words))



if __name__ == "__main__":
    main()
