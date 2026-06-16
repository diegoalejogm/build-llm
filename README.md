# build-llm

Building a GPT-style Large Language Model from scratch in PyTorch, one component at a time.

This repository is a hands-on, notebook-driven implementation that follows the journey of
[Sebastian Raschka's *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch).
Each notebook implements a stage of the pipeline — from raw text tokenization all the way to a
fine-tuned, instruction-following model — with the core building blocks (tokenizer dataloaders,
attention, the transformer block, the GPT model, training loop, and fine-tuning) written by hand
rather than imported from a high-level library.

## Notebooks

The notebooks are meant to be read and run roughly in this order:

| # | Notebook | What it covers |
|---|----------|----------------|
| 1 | [`data-processing.ipynb`](data-processing.ipynb) | Building a `Dataset`/`DataLoader` over text using Byte-Pair Encoding (BPE) tokenization, and creating token + positional embedding layers. |
| 2 | [`coding-attention-mechanisms.ipynb`](coding-attention-mechanisms.ipynb) | Self-attention, causal (masked) self-attention with dropout, and multi-head attention — including an efficient single-matrix implementation. |
| 3 | [`gpt-model.ipynb`](gpt-model.ipynb) | Assembling the full GPT-2 architecture: layer normalization, GELU feed-forward networks, transformer blocks, and shortcut connections. |
| 4 | [`pretrain.ipynb`](pretrain.ipynb) | Text generation, cross-entropy loss, the training loop, evaluation, and loading pretrained GPT-2 weights. |
| 5 | [`fine-tuning.ipynb`](fine-tuning.ipynb) | Fine-tuning the model for classification (SMS spam detection) by adding a classification head. |
| 6 | [`instruction-tuning.ipynb`](instruction-tuning.ipynb) | Instruction tuning on an instruction–response dataset (Alpaca-style prompt formatting). |

## Data

- [`the-verdict.txt`](the-verdict.txt) — the short story used as the corpus for tokenization and pretraining experiments.
- [`sms_spam_collection/`](sms_spam_collection/) — the SMS Spam Collection dataset used for the classification fine-tuning notebook.

Generated artifacts such as [`loss-plot.pdf`](loss-plot.pdf) and [`accuracy-plot.pdf`](accuracy-plot.pdf)
contain training/evaluation curves produced by the notebooks.

## Getting started

This project targets **Python 3.10** (see [`.python-version`](.python-version)) and uses
**PyTorch** for the model, **tiktoken** for BPE tokenization, **TensorFlow/Keras** for loading
the original GPT-2 checkpoint weights, and **matplotlib** for plotting.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and open a notebook
jupyter notebook
```

Then work through the notebooks in the order listed above.

## Project layout

```
build-llm/
├── data-processing.ipynb             # Tokenization, dataloaders, embeddings
├── coding-attention-mechanisms.ipynb # Attention mechanisms
├── gpt-model.ipynb                   # Full GPT-2 architecture
├── pretrain.ipynb                    # Training loop & pretrained weights
├── fine-tuning.ipynb                 # Classification fine-tuning
├── instruction-tuning.ipynb          # Instruction tuning
├── the-verdict.txt                   # Text corpus
├── sms_spam_collection/              # Spam classification dataset
├── requirements.txt
└── .python-version
```

## Acknowledgements

The structure and concepts follow Sebastian Raschka's *Build a Large Language Model (From Scratch)*.
This repository is a personal learning implementation of the ideas presented there.
