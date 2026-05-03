# EduLang Translation Model — Technical Documentation

## Overview

EduLang uses a **Sequence-to-Sequence (Seq2Seq) neural machine translation model** with **Bahdanau (additive) attention** to translate English sentences into French. The model is built with PyTorch and served via a Flask API that the Express web application calls at runtime.

---

## Architecture

### High-Level Diagram

```
English Sentence
       |
   [Tokenize & Encode]
       |
   Encoder (Bidirectional LSTM)
       |
   Encoder Outputs + Hidden/Cell States
       |                    |
   [Attention Mechanism]    |
       |                    |
   Context Vector --------->|
       |
   Decoder (LSTM)
       |
   French Sentence
```

### Components

#### 1. Encoder (`seq2seq.py :: Encoder`)

| Property | Value |
|----------|-------|
| Type | Bidirectional LSTM |
| Layers | 2 |
| Embedding Dim | 256 |
| Hidden Dim | 512 per direction (1024 total) |
| Dropout | 0.3 |
| Input Vocab | 15,004 tokens (15,000 words + PAD, UNK, SOS, EOS) |

**How it works:**
1. The input English sentence is converted to token IDs and passed through an **embedding layer** (15,004 x 256).
2. The embeddings are fed into a **2-layer bidirectional LSTM**. Each direction processes the sentence forward and backward, producing outputs of dimension `hid_dim * 2 = 1024` per timestep.
3. The forward and backward hidden/cell states are **concatenated** and passed through **linear projection layers** (`fc_hidden`, `fc_cell`) with `tanh` activation to compress from 1024 back to 512 dimensions. This bridges the bidirectional encoder to the unidirectional decoder.
4. Returns:
   - `encoder_outputs`: `[batch, src_len, 1024]` — used by attention at every decoder step
   - `hidden`: `[2, batch, 512]` — initial decoder hidden state
   - `cell`: `[2, batch, 512]` — initial decoder cell state

#### 2. Attention Mechanism (`seq2seq.py :: Attention`)

| Property | Value |
|----------|-------|
| Type | Bahdanau (Additive) Attention |
| Input | Decoder hidden state (512) + Encoder outputs (1024) |
| Output | Attention weights over source sequence |

**How it works:**
1. At each decoder timestep, the **top-layer decoder hidden state** (512-dim) is concatenated with every encoder output (1024-dim) to form a 1536-dim vector per source position.
2. This is passed through a linear layer (`attn`: 1536 -> 512) with `tanh` activation, producing an energy score.
3. A second linear layer (`v`: 512 -> 1) reduces each energy to a scalar.
4. Softmax normalizes the scores across all source positions, producing **attention weights**.
5. The weights are used to compute a **context vector** — a weighted sum of encoder outputs — which tells the decoder which parts of the source sentence to focus on.

**Why this matters:** Without attention, the decoder must compress the entire source sentence into a single fixed-size vector. Attention allows the decoder to "look back" at specific source words at each step, dramatically improving translation quality for longer sentences.

#### 3. Decoder (`seq2seq.py :: Decoder`)

| Property | Value |
|----------|-------|
| Type | LSTM |
| Layers | 2 |
| Embedding Dim | 256 |
| Hidden Dim | 512 |
| Dropout | 0.3 |
| Output Vocab | 20,004 tokens (20,000 words + PAD, UNK, SOS, EOS) |

**How it works:**
1. At each timestep, the **previous output token** (or teacher-forced target) is embedded (20,004 x 256).
2. The embedding is **concatenated with the context vector** from attention (256 + 1024 = 1280) and fed into the 2-layer LSTM.
3. The LSTM output, context vector, and embedding are **concatenated** (512 + 1024 + 256 = 1792) and passed through the **output linear layer** (1792 -> 20,004) to produce a probability distribution over the French vocabulary.
4. During training, **teacher forcing** (ratio 0.5) randomly decides whether to use the true target token or the model's own prediction as input for the next step.
5. During inference, **greedy decoding** always uses the model's top prediction (argmax).

#### 4. Seq2Seq Wrapper (`seq2seq.py :: Seq2Seq`)

Orchestrates the encoder and decoder:
1. Encodes the full source sentence.
2. Initializes the decoder with `<SOS>` token.
3. Loops for `trg_len` steps, collecting output predictions.
4. Applies teacher forcing during training.

---

## Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | 61,895,204 |
| Trainable Parameters | 61,895,204 |
| Model File Size | ~743 MB |
| Checkpoint Format | PyTorch `.pth` (state dicts + hyperparams) |

### Parameter Breakdown (approximate)

| Component | Parameters |
|-----------|-----------|
| Encoder Embedding | 15,004 x 256 = ~3.8M |
| Encoder LSTM (2-layer, bidirectional) | ~10.5M |
| Encoder Projection (fc_hidden + fc_cell) | ~1.0M |
| Attention Layers | ~0.8M |
| Decoder Embedding | 20,004 x 256 = ~5.1M |
| Decoder LSTM (2-layer) | ~9.2M |
| Decoder Output Linear | 1,792 x 20,004 = ~35.8M |

---

## Data Pipeline

### Source Data

The training data comes from the **Tatoeba Project** — a collection of parallel English-French sentence pairs (`fra.txt`).

### Processing Steps

```
ml/data/en-fr/fra.txt                  (raw Tatoeba data)
        |
   clean_en_fr.py                       (lowercase, remove special chars,
        |                                filter by length 2-30 words, dedupe)
        v
ml/data/en-fr/clean/en_fr_clean.txt    (238,136 clean sentence pairs)
        |
   split_data.py                        (80/20 random split)
        |
   +--> ml/processed/train.txt          (190,508 pairs)
   +--> ml/processed/val.txt            (47,628 pairs)
        |
   build_vocab.py                       (frequency-based vocab extraction)
        |
   +--> ml/processed/en_vocab.txt       (top 15,000 English words)
   +--> ml/processed/fr_vocab.txt       (top 20,000 French words)
        |
   encode_data.py                       (integer-encode with special tokens)
        |
   +--> ml/processed/train_encoded.txt  (JSON arrays of token IDs)
   +--> ml/processed/val_encoded.txt
```

### Vocabulary

| Language | Vocab Size | Special Tokens |
|----------|-----------|----------------|
| English | 15,004 | `<PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3` |
| French | 20,004 | `<PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3` |

French has a larger vocabulary because French has more word forms due to gendered nouns, verb conjugations, and accented characters.

### Cleaning Rules (`clean_en_fr.py`)

- Lowercase all text
- Remove characters outside `[a-zA-Z, French accented chars, basic punctuation]`
- Collapse whitespace
- Filter out sentences with fewer than 2 or more than 30 words
- Deduplicate identical pairs

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0003 |
| Loss Function | CrossEntropyLoss (ignore_index=0 for padding) |
| Batch Size | 32 |
| Max Epochs | 50 |
| Teacher Forcing Ratio | 0.5 |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping Patience | 5 epochs |

### Training Results

| Epoch | Train Loss | Val Loss | BLEU |
|-------|-----------|----------|------|
| 1 | 3.5225 | 2.8774 | 24.69 |
| 2 | 2.2582 | 2.5768 | — |
| 3 | 1.8350 | 2.4741 | — |
| 4 | 1.5913 | 2.4277 | — |
| 5 | 1.4220 | 2.4080 | 36.81 |
| **6** | **1.2959** | **2.3945** | — |
| 7 | 1.1970 | 2.4330 | — |
| 8 | 1.1154 | 2.4459 | — |
| 9 | 1.0534 | 2.4669 | — |
| 10 | 0.9948 | 2.4865 | 40.02 |
| 11 | 0.9536 | 2.4957 | — |

- **Best checkpoint saved at epoch 6** (lowest val loss: 2.3945)
- **Early stopping triggered at epoch 11** (5 epochs with no val loss improvement)
- **Final BLEU on best model: 37.71**
- Training completed on **CUDA GPU**

### Early Stopping

The model monitors validation loss after each epoch. If val loss does not improve for 5 consecutive epochs, training stops and the best checkpoint (lowest val loss) is retained. This prevents overfitting — as shown above, train loss kept decreasing while val loss started rising after epoch 6.

### BLEU Evaluation

BLEU (Bilingual Evaluation Understudy) is computed every 5 epochs using:
- **Corpus-level BLEU** with n-grams up to 4
- **Brevity penalty** for translations shorter than references
- **Greedy decoding** (argmax at each step, max 50 tokens)

---

## Inference

### How Translation Works (`infer.py`)

1. **Input**: English sentence string
2. **Tokenize**: Lowercase, split by whitespace
3. **Encode**: Map words to vocab IDs, prepend `<SOS>`, append `<EOS>`. Unknown words become `<UNK>`.
4. **Encoder forward pass**: Produces encoder outputs and initial decoder states
5. **Decoder loop** (max 30 steps):
   - Compute attention over encoder outputs
   - Generate next French token
   - Stop if `<EOS>` is predicted
6. **Decode**: Map predicted IDs back to French words
7. **Output**: French sentence string

### API Server (`api.py`)

The Flask server loads the model once at startup and serves translations via HTTP:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /translate` | POST | Translate text |
| `GET /health` | GET | Health check |

**Request format:**
```json
{
  "text": "the weather is nice today",
  "from": "english",
  "to": "french"
}
```

**Response format:**
```json
{
  "original": "the weather is nice today",
  "translated": "il fait beau aujourd'hui",
  "from": "english",
  "to": "french"
}
```

### Translation Examples

| English | French (Model Output) | Correct |
|---------|----------------------|---------|
| hello how are you | salut, comment allez-vous? | Yes |
| i love you | je t'aime | Yes |
| the cat is on the table | le chat est sur la table | Yes |
| what is your name | quel est votre nom ? | Yes |
| she is very beautiful | elle est tres belle | Yes |
| we are going to school | nous allons a l'ecole | Yes |
| he wants to eat an apple | il veut manger une pomme | Yes |
| the weather is nice today | il fait beau aujourd'hui | Yes |
| i do not understand french | je ne comprends pas le francais | Yes |

---

## File Structure

```
ml/
├── data/
│   ├── en-fr/
│   │   ├── fra.txt                    # Raw Tatoeba EN-FR parallel corpus
│   │   ├── _about.txt                 # Dataset info
│   │   └── clean/
│   │       └── en_fr_clean.txt        # Cleaned sentence pairs
│   └── en-es/
│       ├── spa.txt                    # Raw EN-ES data (unused)
│       └── _about.txt
│
├── processed/
│   ├── en_vocab.txt                   # English vocabulary (15,000 words)
│   ├── fr_vocab.txt                   # French vocabulary (20,000 words)
│   ├── train.txt                      # Training pairs (190,508)
│   ├── val.txt                        # Validation pairs (47,628)
│   ├── train_encoded.txt              # Integer-encoded training data
│   └── val_encoded.txt                # Integer-encoded validation data
│
├── model/
│   ├── seq2seq.py                     # Model architecture (Encoder, Attention, Decoder, Seq2Seq)
│   ├── dataloader.py                  # Dataset class and batch padding
│   ├── train.py                       # Training loop with validation and early stopping
│   ├── infer.py                       # CLI inference script
│   ├── api.py                         # Flask API server
│   └── edulang_seq2seq.pth            # Trained model checkpoint (~743 MB)
│
└── scripts/
    ├── clean_en_fr.py                 # Step 1: Clean raw data
    ├── split_data.py                  # Step 2: Train/val split
    ├── build_vocab.py                 # Step 3: Build vocabularies
    ├── tokenize.py                    # Utility: Inspect vocab statistics
    └── encode_data.py                 # Step 4: Integer-encode sentences
```

---

## How to Retrain

### Prerequisites
- Python 3.11+
- PyTorch 2.x (CUDA recommended)
- Flask

### Steps

```bash
# 1. Clean the raw data
python ml/scripts/clean_en_fr.py

# 2. Split into train/val
python ml/scripts/split_data.py

# 3. Build vocabularies
python ml/scripts/build_vocab.py

# 4. Encode sentences as integer arrays
python ml/scripts/encode_data.py

# 5. Train the model
cd ml/model
python train.py
```

Training will:
- Use all available data (no cap)
- Log train loss, val loss, and BLEU scores
- Save the best model automatically
- Stop early if val loss plateaus for 5 epochs

### Adjusting Hyperparameters

Edit the constants at the top of `ml/model/train.py`:

```python
INPUT_DIM = 15004       # Must match en_vocab size + 4 special tokens
OUTPUT_DIM = 20004      # Must match fr_vocab size + 4 special tokens
EMB_DIM = 256           # Embedding dimensions
HID_DIM = 512           # LSTM hidden dimensions
N_LAYERS = 2            # Number of LSTM layers
DROPOUT = 0.3           # Dropout rate
EPOCHS = 50             # Maximum epochs
BATCH_SIZE = 32         # Batch size (reduce if OOM)
LR = 0.0003             # Learning rate
PATIENCE = 5            # Early stopping patience
```

---

## Limitations

1. **English to French only** — The model is trained unidirectionally. French-to-English requires a separate model or a bidirectional training setup.
2. **Whitespace tokenization** — Cannot handle unseen word forms (e.g., new conjugations, compound words). Subword tokenization (BPE/SentencePiece) would improve this.
3. **Greedy decoding** — Always picks the single most likely token. Beam search (k=3-5) would produce better translations.
4. **No attention to input length** — Very long sentences (>30 words) may degrade in quality since training data was capped at 30 words.
5. **`<UNK>` for rare words** — Words outside the top 15K English or 20K French are mapped to `<UNK>`.
6. **No GPU required for inference** — The model runs on CPU for inference, but training benefits significantly from CUDA.

## Future Improvements

- **Beam search decoding** — Try top-k candidates at each step for better translations
- **Subword tokenization** — Use SentencePiece/BPE to eliminate most `<UNK>` tokens
- **Transformer architecture** — Replace LSTM with self-attention for better parallelism and quality
- **Bidirectional translation** — Train FR->EN alongside EN->FR
- **More language pairs** — Spanish data (`en-es/spa.txt`) is already present but unused
