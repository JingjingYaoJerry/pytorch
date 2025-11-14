## LeNET--CIFAR-10 (mixed PyTorch & tooling examples)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-optional-orange.svg)](https://huggingface.co/docs/transformers)

This repository is a collection of small, focused example scripts and utilities for learning and prototyping in deep learning, data preprocessing, and Hugging Face workflows. The codebase includes:

- A LeNet-style CNN example and CIFAR-10 demo (`cnn.py`).
- Several interactive Hugging Face/Transformers helper scripts (`huggingface.py`, `huggingface_2.py`, `huggingface_3_5.py`) that demonstrate pipelines, tokenizer+model flows, and Trainer/custom training loops.
- Data preprocessing utilities and interactive/GUI cleaners (`preprocessing.py`, `preprocessing_template.py`, `preprocessing_review.py`, `preprocess_strict.py`-style implementations).
- A small mushroom classification preprocessing scaffold (`mushroom_classification.py`).
- Tiny helpers and environment checks (`model.py`).
- Sample datasets under `data/` (CIFAR-10 batches and `mushrooms.csv`) for local experiments.

The intent is educational: show runnable examples, explain common workflows (data → model → evaluation), and provide small building blocks you can adapt for experiments or interview demos.

---

### Key features (by area)

Vision / CNN

* `cnn.py` — LeNet-style convolutional network adapted for CIFAR-10 (3-channel inputs). Includes data transforms, DataLoader usage, training loop (SGD + CrossEntropyLoss), simple logging and final test evaluation.

Hugging Face / NLP / Multimodal demos

* `huggingface.py`, `huggingface_2.py`, `huggingface_3_5.py` — interactive CLI wrappers that demonstrate:
  - high-level `pipeline` use for tasks (text-generation, QA, image-classification, etc.),
  - tokenizer + model manual workflows (AutoTokenizer/AutoModel variants),
  - Trainer-based and custom PyTorch training loops with dataset/tokenization examples, and optional `datasets`/`evaluate`/`wandb` integration.

Data preprocessing

* `preprocessing_template.py`, `preprocessing_review.py`, and `preprocessing.py` — examples of robust CSV loading, cleaning steps (duplicates, missing, inf, outliers), and interactive CLI/Tkinter-based review flows for manual inspection and guided fixes.

Other

* `mushroom_classification.py` — a scaffold that demonstrates loading `data/mushrooms.csv`, simple feature encoding (OneHotEncoder), and dataset splitting for classification experiments.
* `model.py` — tiny helper that checks the available device and prints environment info.

---

### Quick setup & dependencies (PowerShell)

These examples assume Python 3.8+. Install core dependencies as needed; some scripts are optional and require `transformers`, `datasets`, or `wandb`.

1. Clone the repo and open the folder:

```powershell
git clone https://github.com/JingjingYaoJerry/LeNET--CIFAR-10.git
cd "LeNET--CIFAR-10"
```

2. (Recommended) create & activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install minimal packages for the PyTorch/CNN demo (adjust versions for your platform):

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib pandas scikit-learn
```

4. Optional (Transformers / datasets / evaluate / wandb):

```powershell
python -m pip install transformers datasets evaluate huggingface_hub wandb
```

Notes: some scripts are interactive and will prompt for checkpoints or API keys. For Transformer demos that use Hub models, set environment variables or a `.env` file containing keys if required by the model provider.

---

### How to run (examples — PowerShell)

Run the CIFAR-10 LeNet demo (small, downloads dataset if missing):

```powershell
python .\cnn.py
```

Run an interactive Hugging Face demo (follows prompts):

```powershell
python .\huggingface_2.py
# or
python .\huggingface_3_5.py
```

Inspect / run preprocessing review CLI:

```powershell
python .\preprocessing_review.py
```

Run the mushroom dataset scaffold to inspect preprocessing steps:

```powershell
python .\mushroom_classification.py
```

---

### Project structure

```
LeNET--CIFAR-10/
├── cnn.py                       # LeNet example & CIFAR-10 training loop
├── model.py                     # small device helper
├── huggingface.py               # minimal HF interactive wrapper
├── huggingface_2.py             # richer CLI demo (pipeline/tokenizer+model)
├── huggingface_3_5.py           # Trainer/custom training loop examples
├── preprocessing.py             # helper snippets and robust CSV utilities
├── preprocessing_template.py    # template notes about common cleaning cases
├── preprocessing_review.py      # interactive CLI review for data cleaning
├── mushroom_classification.py   # simple dataset preprocessing scaffold
├── README.md
└── data/
    ├── mushrooms.csv
    └── cifar-10-batches-py/     # (optionally add downloaded CIFAR-10 files here)
```

---

### Tips & caveats

- Several scripts are designed as educational, interactive demos — they prompt for input or open a small GUI (Tkinter) for reviewing data. They are not production-ready pipelines.
- Some code references `torch.accelerator` or optional packages: if you encounter import errors, run the minimal install commands above or use CPU-only installs of PyTorch.
- If you plan to use Transformer demos with private API keys or large models, follow Hugging Face guidance for `huggingface_hub` authentication and set `HF_HOME` / environment variables as needed.

---

### Acknowledgements

This repository is a personal collection of learning examples and demos by Jingjing YAO (Jerry). It re-uses and demonstrates third-party libraries including PyTorch, torchvision, Hugging Face Transformers, datasets, and Qdrant in separate projects. See individual scripts for license and citation notes.
