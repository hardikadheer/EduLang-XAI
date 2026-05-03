def load_vocab(path):
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    idx = 4
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def encode_sentence(sentence, vocab):
    encoded = [vocab.get("<SOS>")]
    for word in sentence.split():
        encoded.append(vocab.get(word, vocab["<UNK>"]))
    encoded.append(vocab.get("<EOS>"))
    return encoded


def encode_file(input_path, output_path, en_vocab, fr_vocab):
    import json
    with open(input_path, "r", encoding="utf-8") as f, \
         open(output_path, "w", encoding="utf-8") as out:
        for line in f:
            en, fr = line.strip().split("\t")
            en_encoded = encode_sentence(en, en_vocab)
            fr_encoded = encode_sentence(fr, fr_vocab)
            out.write(f"{json.dumps(en_encoded)}\t{json.dumps(fr_encoded)}\n")


# paths
en_vocab_path = "ml/processed/en_vocab.txt"
fr_vocab_path = "ml/processed/fr_vocab.txt"

train_path = "ml/processed/train.txt"
val_path = "ml/processed/val.txt"

train_out = "ml/processed/train_encoded.txt"
val_out = "ml/processed/val_encoded.txt"

# run
en_vocab = load_vocab(en_vocab_path)
fr_vocab = load_vocab(fr_vocab_path)

encode_file(train_path, train_out, en_vocab, fr_vocab)
encode_file(val_path, val_out, en_vocab, fr_vocab)

print("Encoding complete")
print("English vocab size:", len(en_vocab))
print("French vocab size:", len(fr_vocab))
