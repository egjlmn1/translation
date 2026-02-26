# 🌍 EN↔FR Translation Platform

A full-stack, from-scratch English-to-French translation ecosystem. This project implements a complete pipeline including a custom BPE tokenizer, a Transformer-based neural engine, and a real-time monitoring dashboard.

## 🚀 Overview

This platform is built across **25 files** organized into 7 specialized packages. It handles everything from raw data streaming to serving a "Google Translate-style" web interface.

### Key Components

* 
**Custom Tokenizer:** A Byte Pair Encoding (BPE) implementation capable of learning 32K merge rules from up to 1M rows of text.


* 
**Transformer Model:** A core `nn.Transformer` architecture initialized from scratch to handle sequence-to-sequence translation.


* 
**Streaming Data Pipeline:** Efficiently processes an 8.4 GB dataset using chunked CSV streaming to keep memory usage low.


* 
**Live Dashboard:** A dark-themed web UI that provides real-time training metrics, BLEU scores, and sample predictions via WebSockets.



---

## 🛠️ Project Structure

```text
Translation/
[cite_start]├── config.yaml          # Central hyperparameter management [cite: 1, 12]
[cite_start]├── tokenizer/           # BPE train, encode, and decode logic [cite: 1]
[cite_start]├── model/               # Transformer encoder-decoder implementation [cite: 1, 2]
[cite_start]├── data/                # Chunked CSV streaming dataset loader [cite: 1, 13]
[cite_start]├── training/            # Training loops, checkpointing, and BLEU metrics [cite: 1, 14]
[cite_start]├── api/                 # FastAPI server and inference engine [cite: 1, 11]
[cite_start]├── web/                 # UIs for training (dashboard) and translation [cite: 1, 3, 15]
[cite_start]└── scripts/             # CLI entry points for all operations [cite: 1]

```

---

## ⚡ Quick Start

### 1. Environment Setup

Install the necessary Python dependencies:

```bash
pip install -r requirements.txt

```

### 2. Tokenizer Training

Prepare the BPE merge rules using your source dataset:

```bash
python scripts/train_tokenizer.py

```

### 3. Model Training

Start the training process with a live dashboard to monitor performance:

```bash
python scripts/train_model.py --with-server

```

> 
> **Tip:** To test the pipeline quickly, set `max_rows: 10000` in `config.yaml` to run on a smaller data subset.
> 
> 

### 4. Run the Translation UI

Launch the standalone inference server to translate text:

```bash
python scripts/serve.py

```

* 
**Translation UI:** `http://localhost:8000/translate` 


* 
**Metrics Dashboard:** `http://localhost:8000/dashboard` 
