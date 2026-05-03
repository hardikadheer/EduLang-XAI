import torch
import os
from torch.utils.data import DataLoader
from collections import Counter

from seq2seq import Encoder, Decoder, Seq2Seq
from dataloader import TranslationDataset, pad_batch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "edulang_seq2seq.pth")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

encoder = Encoder(
    checkpoint["input_dim"],
    checkpoint["emb_dim"],
    checkpoint["hid_dim"],
    checkpoint["n_layers"],
    checkpoint["dropout"]
)

decoder = Decoder(
    checkpoint["output_dim"],
    checkpoint["emb_dim"],
    checkpoint["hid_dim"],
    checkpoint["n_layers"],
    checkpoint["dropout"]
)

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

model.eval()


val_data = TranslationDataset(os.path.join(PROCESSED_DIR, "val_encoded.txt"))
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=pad_batch)


val_loader = list(val_loader)[:2]   


def compute_bleu(references, hypotheses, max_n=4):
    clipped_counts = Counter()
    total_counts = Counter()
    ref_len, hyp_len = 0, 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))

            for ngram in hyp_ngrams:
                clipped_counts[n] += min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0))
                total_counts[n] += hyp_ngrams[ngram]

    if hyp_len == 0:
        return 0.0

    import math
    bp = math.exp(min(0, 1 - ref_len / hyp_len))

    log_avg = 0
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            return 0.0
        log_avg += math.log((clipped_counts[n] + 1e-9) / total_counts[n])
    log_avg /= max_n

    return bp * math.exp(log_avg) * 100



def greedy_decode(model, src, max_len=15, sos_idx=2, eos_idx=3):  # 🔥 reduced length
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        batch_size = src.shape[0]
        input_tok = torch.full((batch_size,), sos_idx, dtype=torch.long).to(src.device)

        outputs = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_tok, hidden, cell, encoder_outputs)
            preds = output.argmax(1)
            input_tok = preds

            for i in range(batch_size):
                if not finished[i]:
                    token = preds[i].item()
                    if token == eos_idx:
                        finished[i] = True
                    else:
                        outputs[i].append(token)

            if all(finished):
                break

    return outputs

def load_vocab(path):
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    idx = 4
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def invert_vocab(vocab):
    return {v: k for k, v in vocab.items()}

fr_vocab = load_vocab(os.path.join(PROCESSED_DIR, "fr_vocab.txt"))
idx2word = invert_vocab(fr_vocab)  

def tokens_to_sentence(tokens, idx2word):
    words = []
    for t in tokens:
        if t in (0, 2, 3):  # skip <PAD>, <SOS>, <EOS>
            continue
        words.append(idx2word.get(t, "<UNK>"))
    return " ".join(words)

def evaluate_bleu(model, loader):
    references, hypotheses = [], []

    for i, (src, trg) in enumerate(loader):
        if i == 2:   
            break

        print(f"Processing batch {i+1}...")

        src, trg = src.to(DEVICE), trg.to(DEVICE)

        preds = greedy_decode(model, src)

        for j in range(trg.shape[0]):
            ref = trg[j].tolist()
            ref = [t for t in ref if t not in (0, 2, 3)]

           
            pred = [t for t in preds[j] if t not in (0, 2, 3)]

            
            references.append(ref)
            hypotheses.append(pred)

            print("\nReference:", ref)
            print("Prediction:", pred)

    return compute_bleu(references, hypotheses)


if __name__ == "__main__":
    bleu = evaluate_bleu(model, val_loader)
    print(f"\n BLEU Score: {bleu:.2f}")


ref_sentence = tokens_to_sentence(ref, idx2word)
pred_sentence = tokens_to_sentence(pred, idx2word)

print("\nReference:", ref_sentence)
print("Prediction:", pred_sentence)