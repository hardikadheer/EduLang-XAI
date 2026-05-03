from collections import Counter

train_path = "ml/processed/train.txt"

en_vocab = Counter()
fr_vocab = Counter()

with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        en, fr = line.strip().split("\t")
        en_vocab.update(en.split())
        fr_vocab.update(fr.split())

print("Top English words:", en_vocab.most_common(10))
print("Top French words:", fr_vocab.most_common(10))
print("English vocab size:", len(en_vocab))
print("French vocab size:", len(fr_vocab))
