from collections import Counter

train_path = "ml/processed/train.txt"

MAX_EN = 15000
MAX_FR = 20000

en_vocab = Counter()
fr_vocab = Counter()

with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        en, fr = line.strip().split("\t")
        en_vocab.update(en.split())
        fr_vocab.update(fr.split())

en_words = [w for w, _ in en_vocab.most_common(MAX_EN)]
fr_words = [w for w, _ in fr_vocab.most_common(MAX_FR)]

with open("ml/processed/en_vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(en_words))

with open("ml/processed/fr_vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(fr_words))

print("English vocab saved:", len(en_words))
print("French vocab saved:", len(fr_words))

