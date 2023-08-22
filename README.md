This is a repository for the CUNI submission to the [MRL 2023 Shared Task on
Multi-lingual Multi-task Information
Retrieval](https://sigtyp.github.io/st2023-mrl.html).

The submission takes the translate-test approach, it translates the test data
into English, solves the tasks in English and projects the predicted tags back
into English.

# How to run the experiments

## TL;DR

We use [Snakemake](https://snakemake.readthedocs.io/en/stable/) (a Python
analogue of Makefiles) to manage the experiments. Running the `snakemake`
commands in the `ner` and `qa` directories should run experiments.

## Prerequisites

Due to inconsistent requirements of the tools we use, we use two virtual
environments. The default environment is specified in `requirements.txt` (run
`pip install -R requirements.txt`) to install it.

Transformers-NER work with an older version of PyTorch and Transformers and
requires its own virtual environment.

```bash
pip install tner
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

# Additional Resources used in the Shared Task

The shared task is built around the [XTREME-UP
dataset](https://github.com/google-research/xtreme-up), but external tools and
data sources are allowed. In our case, it were the following:

* Machine Translation: "facebook/nllb-200-3.3B", "ychenNLP/nllb-200-3.3B-easyproject"
* NER, t-NER models: "tner/roberta-large-ontonotes5", "tner/roberta-large-conll2003",
* Sentence splitting, WTPSplit: "wtp-canine-s-12l",
* Question answering: "deepset/roberta-large-squad2"
