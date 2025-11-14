# AI Engineering Playground

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Made-with-Love](https://img.shields.io/badge/Made%20with-❤️-ff69b4.svg)](https://www.linkedin.com/in/%E4%BA%AC%E6%99%B6-%E5%A7%9A-9997b5180/)

A consolidated sandbox for practicing real-world machine learning workflows.
This repo implements and extends key components from the official PyTorch and Hugging Face tutorials, adding engineering utilities, interactive CLIs, and experiment-friendly modules.

It is designed as a hands-on environment for learning, debugging, and testing small ML workflows — without the heavy overhead of full projects.

---

### Features

Hugging Face

* `huggingface.py`, `huggingface_2.py`, `huggingface_3_5.py` — interactive CLI wrappers that demonstrate the official tutorial docs:
  - high-level `pipeline` use for tasks (text-generation, QA, image-classification, etc.),
  - tokenizer + model manual workflows (AutoTokenizer/AutoModel variants),
  - Trainer-based and custom PyTorch training loops with dataset/tokenization examples, and optional `datasets`/`evaluate`/`wandb` integration.

Data Preprocessing

* `preprocessing_template.py`, `preprocessing_CLI.py`, and `preprocessing_GUI.py` — examples/steps of data-preprocessing, and interactive CLI/Tkinter-based flows for manual inspection and guided fixes.

---

### Purpose
This repository is a learning-focused ML engineering playground, intended to practice:
* end-to-end data preprocessing
* training loop implementation
* model inspection & debugging
* Hugging Face workflows
* CLI tooling & dynamic utilities
* checkpoint saving & versioning
* dataset handling and batching

It demonstrates the ability to build working ML systems, not just call high-level APIs.

---

### How to run (examples — PowerShell)

Run an interactive Hugging Face helper (follows prompts):

```powershell
python .\huggingface_2.py
# or
python .\huggingface_3_5.py
```

Inspect / run data-preprocessing CLI:

```powershell
python .\preprocessing_CLI.py
```

Run the CIFAR-10 LeNet demo (small, downloads dataset if missing):

```powershell
python .\cnn.py
```

---

### Project structure

```
ai-engineering-playground/
├── huggingface.py               # minimal HF interactive wrapper
├── huggingface_2.py             # richer CLI demo (pipeline/tokenizer+model)
├── huggingface_3_5.py           # Trainer/custom training loop examples
├── preprocessing_template.py             # helper snippets and robust CSV utilities
├── preprocessing_CLI.py    # template notes about common cleaning cases
├── preprocessing_GUI.py      # interactive CLI review for data cleaning
├── cnn.py                       # LeNet example & CIFAR-10 training loop
├── README.md
├── ...
└── data/
    ├── cifar-10-batches-py/
    └── ...
```

---

### Tips & caveats

- Several scripts are designed as educational, interactive demos — they prompt for input or open a small GUI (Tkinter) for reviewing data. They are not production-ready pipelines.

---