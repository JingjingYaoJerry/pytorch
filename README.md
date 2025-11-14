## PyTorch & Hugging Face & Pre-Processing

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Made-with-Love](https://img.shields.io/badge/Made%20with-❤️-ff69b4.svg)](https://www.linkedin.com/in/%E4%BA%AC%E6%99%B6-%E5%A7%9A-9997b5180/)

This repository is a collection of small, focused example scripts and utilities for learning and prototyping in machine learning, data preprocessing, and Hugging Face workflows. The codebase includes:

- A LeNet-style CNN example and CIFAR-10 demo (`cnn.py`).
- Several interactive Hugging Face/Transformers helper scripts that demonstrate pipelines, tokenizer+model flows, and Trainer/custom training loops based on the official tutorial docs.
- Data preprocessing utilities and interactive/GUI cleaners (`preprocessing_template.py`, `preprocessing_CLI.py`, `preprocessing_GUI.py`, `preprocess_strict.py`-style implementations).

The intent is educational and self-review: show runnable examples, explain common workflows (data → model → evaluation), and provide small building blocks for experiments or reflections.

---

### Key features (by area)

PyTorch / CV / CNN

* `cnn.py` — LeNet-style convolutional network adapted for CIFAR-10 (3-channel inputs). Includes data transforms, DataLoader usage, training loop (SGD + CrossEntropyLoss), simple logging and final test evaluation.

Hugging Face / NLP / Transformers

* `huggingface.py`, `huggingface_2.py`, `huggingface_3_5.py` — interactive CLI wrappers that demonstrate the official tutorial docs:
  - high-level `pipeline` use for tasks (text-generation, QA, image-classification, etc.),
  - tokenizer + model manual workflows (AutoTokenizer/AutoModel variants),
  - Trainer-based and custom PyTorch training loops with dataset/tokenization examples, and optional `datasets`/`evaluate`/`wandb` integration.

Data Preprocessing

* `preprocessing_template.py`, `preprocessing_CLI.py`, and `preprocessing_GUI.py` — examples/steps of data-preprocessing, and interactive CLI/Tkinter-based flows for manual inspection and guided fixes.

---

### How to run (examples — PowerShell)

Run the CIFAR-10 LeNet demo (small, downloads dataset if missing):

```powershell
python .\cnn.py
```

Run an interactive Hugging Face helper (follows prompts):

```powershell
python .\huggingface_2.py
# or
python .\huggingface_3_5.py
```

Inspect / run data-preprocessing CLI:

```powershell
python .\preprocessing_review.py
```

---

### Project structure

```
LeNET--CIFAR-10/
├── cnn.py                       # LeNet example & CIFAR-10 training loop
├── huggingface.py               # minimal HF interactive wrapper
├── huggingface_2.py             # richer CLI demo (pipeline/tokenizer+model)
├── huggingface_3_5.py           # Trainer/custom training loop examples
├── preprocessing_template.py             # helper snippets and robust CSV utilities
├── preprocessing_CLI.py    # template notes about common cleaning cases
├── preprocessing_GUI.py      # interactive CLI review for data cleaning
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

### Acknowledgements

This repository is a personal collection of learning examples and demos only. It re-uses and demonstrates third-party libraries including PyTorch, torchvision, Hugging Face Transformers, datasets, etc. in separate projects.