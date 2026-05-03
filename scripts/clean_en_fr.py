import re

input_path = "../data/en-fr/fra.txt"
output_path = "ml/data/en-fr/clean/en_fr_clean.txt"

def clean_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Zàâçéèêëîïôûùüÿñæœ'’.,!? ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

clean_pairs = set()

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if "\t" not in line:
            continue

        en, fr = line.strip().split("\t")[:2]

        en_clean = clean_sentence(en)
        fr_clean = clean_sentence(fr)

        if len(en_clean.split()) < 2 or len(fr_clean.split()) < 2:
            continue
        if len(en_clean.split()) > 30 or len(fr_clean.split()) > 30:
            continue

        clean_pairs.add(f"{en_clean}\t{fr_clean}")

with open(output_path, "w", encoding="utf-8") as f:
    for pair in clean_pairs:
        f.write(pair + "\n")

print(f"Cleaned dataset size: {len(clean_pairs)} sentence pairs")
