import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from seq2seq import Encoder, Decoder, Seq2Seq
from dataloader import TranslationDataset, pad_batch

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --- Hyperparameters ---
INPUT_DIM = 15004
OUTPUT_DIM = 20004
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0003
PATIENCE = 5  # early stopping patience

# --- Model ---
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized — {param_count:,} trainable parameters")

# --- Data (no cap — use all available data) ---
train_data = TranslationDataset(os.path.join(PROCESSED_DIR, "train_encoded.txt"))
val_data = TranslationDataset(os.path.join(PROCESSED_DIR, "val_encoded.txt"))

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_batch
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Batches per epoch: {len(train_loader)}")


# --- BLEU helpers ---
def compute_bleu(references, hypotheses, max_n=4):
    """Compute corpus-level BLEU score (simplified)."""
    clipped_counts = Counter()
    total_counts = Counter()
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter()
            for i in range(len(ref) - n + 1):
                ref_ngrams[tuple(ref[i:i + n])] += 1

            hyp_ngrams = Counter()
            for i in range(len(hyp) - n + 1):
                hyp_ngrams[tuple(hyp[i:i + n])] += 1

            for ngram, count in hyp_ngrams.items():
                clipped_counts[n] += min(count, ref_ngrams.get(ngram, 0))
                total_counts[n] += count

    if hyp_len == 0:
        return 0.0

    # Brevity penalty
    import math
    bp = math.exp(min(0, 1 - ref_len / hyp_len)) if hyp_len > 0 else 0

    # Geometric mean of precisions
    log_avg = 0
    for n in range(1, max_n + 1):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_avg += math.log(clipped_counts[n] / total_counts[n])
    log_avg /= max_n

    return bp * math.exp(log_avg) * 100


def greedy_decode(model, src, max_len=50, sos_idx=2, eos_idx=3):
    """Greedy decode a single batch for BLEU evaluation."""
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        batch_size = src.shape[0]
        input_tok = torch.full((batch_size,), sos_idx, dtype=torch.long).to(src.device)

        all_preds = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_tok, hidden, cell, encoder_outputs)
            preds = output.argmax(1)
            input_tok = preds

            for b in range(batch_size):
                if not finished[b]:
                    tok = preds[b].item()
                    if tok == eos_idx:
                        finished[b] = True
                    else:
                        all_preds[b].append(tok)

            if all(finished):
                break

    return all_preds


def evaluate_bleu(model, data_loader, device):
    """Run greedy decoding on a data loader and compute BLEU."""
    references = []
    hypotheses = []

    for src, trg in data_loader:
        src = src.to(device)
        trg = trg.to(device)

        preds = greedy_decode(model, src)
        hypotheses.extend(preds)

        for i in range(trg.shape[0]):
            ref = trg[i].tolist()
            # Strip PAD(0), SOS(2), EOS(3)
            ref = [t for t in ref if t not in (0, 2, 3)]
            references.append(ref)

    return compute_bleu(references, hypotheses)


# --- Validation loss ---
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# --- Training loop with early stopping ---
best_val_loss = float("inf")
patience_counter = 0
SAVE_PATH = os.path.join(BASE_DIR, "edulang_seq2seq.pth")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for src, trg in train_loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)

    # BLEU every 5 epochs (expensive)
    bleu_str = ""
    if (epoch + 1) % 5 == 0 or epoch == 0:
        bleu = evaluate_bleu(model, val_loader, DEVICE)
        bleu_str = f" | BLEU: {bleu:.2f}"

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}{bleu_str}"
    )

    # Early stopping + best model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "emb_dim": EMB_DIM,
            "hid_dim": HID_DIM,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
        }, SAVE_PATH)
        print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
            break

# Final BLEU on best model
checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
final_bleu = evaluate_bleu(model, val_loader, DEVICE)
print(f"\nTraining complete — Best val loss: {best_val_loss:.4f} | Final BLEU: {final_bleu:.2f}")
print("Model saved at:", SAVE_PATH)

