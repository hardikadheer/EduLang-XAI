import random

input_path = "ml/data/en-fr/clean/en_fr_clean.txt"
train_path = "ml/processed/train.txt"
val_path = "ml/processed/val.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

split = int(0.8 * len(lines))
train_lines = lines[:split]
val_lines = lines[split:]

with open(train_path, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(val_path, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"Train size: {len(train_lines)}")
print(f"Validation size: {len(val_lines)}")
