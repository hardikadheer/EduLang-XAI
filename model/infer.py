import torch
import os
import warnings
warnings.filterwarnings("ignore")

from seq2seq import Encoder, Decoder, Seq2Seq
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "edulang_seq2seq.pth")


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


en_vocab = load_vocab(os.path.join(PROCESSED_DIR, "en_vocab.txt"))
fr_vocab = load_vocab(os.path.join(PROCESSED_DIR, "fr_vocab.txt"))
fr_ivocab = invert_vocab(fr_vocab)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

n_layers = checkpoint.get("n_layers", 2)
dropout = checkpoint.get("dropout", 0.3)

encoder = Encoder(
    checkpoint["input_dim"],
    checkpoint["emb_dim"],
    checkpoint["hid_dim"],
    n_layers,
    dropout
)

decoder = Decoder(
    checkpoint["output_dim"],
    checkpoint["emb_dim"],
    checkpoint["hid_dim"],
    n_layers,
    dropout
)

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
model.eval()


def encode_sentence(sentence, vocab):
    tokens = sentence.lower().strip().split()
    print("\n[TOKENS]:", tokens)

    indices = [vocab["<SOS>"]]
    for tok in tokens:
        indices.append(vocab.get(tok, vocab["<UNK>"]))

    indices.append(vocab["<EOS>"])
    print("[INDICES]:", indices)

    tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE)
    print("[INPUT TENSOR]:", tensor)

    return tensor, tokens



def translate(sentence, max_len=30):
    src, input_tokens = encode_sentence(sentence, en_vocab)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        print("\n========== ENCODER STATES ==========")

        print(f"[HIDDEN STATE SHAPE]: {hidden.shape}")
        print(f"[CELL STATE SHAPE]: {cell.shape}")

        # Print small sample (avoid full tensor)
        print("[HIDDEN STATE SAMPLE]:", hidden[0][0][:5].tolist())
        print("[CELL STATE SAMPLE]:", cell[0][0][:5].tolist())

        input_tok = torch.tensor([fr_vocab["<SOS>"]]).to(DEVICE)
        result_ids = []

        print("\n========== DECODING STEPS ==========")

        for step in range(max_len):
            output, hidden, cell = model.decoder(input_tok, hidden, cell, encoder_outputs)

            pred = output.argmax(1).item()
            print(f"Step {step+1} → Token ID:", pred)

            if pred == fr_vocab["<EOS>"]:
                print("→ <EOS> reached, stopping.")
                break

            result_ids.append(pred)
            input_tok = torch.tensor([pred]).to(DEVICE)

    result_tokens = [fr_ivocab.get(idx, "<UNK>") for idx in result_ids]

    print("\n[PREDICTED TOKEN IDS]:", result_ids)
    print("[PREDICTED TOKENS]:", result_tokens)

    output_sentence = " ".join(result_tokens)
    print("[OUTPUT SENTENCE]:", output_sentence)

    print("\n========== N-GRAMS ==========")
    for n in [1, 2, 3]:
        grams = list(ngrams(result_tokens, n))
        print(f"\n{n}-grams:")
        for g in grams:
            print(g)

    return output_sentence


if __name__ == "__main__":
    print("\nEnglish -> French Translator (EduLang)")
    print("Type 'q' to quit")

    while True:
        sentence = input("\nEnter English sentence: ")
        if sentence.lower() == "q":
            break

        print("\n================ PROCESSING =================")

        predicted = translate(sentence)

        print("\nFrench (Predicted):", predicted)

        reference = input("\nEnter correct French sentence (for BLEU): ")

        ref_tokens = reference.lower().split()
        pred_tokens = predicted.lower().split()

        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)

        print("\nBLEU Score:", round(bleu * 100, 2), "%")