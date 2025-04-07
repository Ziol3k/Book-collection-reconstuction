import os
import random

# Ścieżki
processed_folder = "../data/processed/lines/"
split_file_path = "../data/raw/ascii/train-test-split.txt"

# Lista plików w folderze
files = [f.split(".")[0] for f in os.listdir(processed_folder) if f.endswith(".png")]

# Ustawienia proporcji
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Losowy podział danych
random.shuffle(files)
train_count = int(len(files) * train_ratio)
test_count = int(len(files) * test_ratio)

train_files = files[:train_count]
test_files = files[train_count:train_count + test_count]
val_files = files[train_count + test_count:]

# Zapis podziału do pliku
with open(split_file_path, "w") as f:
    for file in train_files:
        f.write(f"{file} train\n")
    for file in test_files:
        f.write(f"{file} test\n")
    for file in val_files:
        f.write(f"{file} val\n")

print(f"Podział danych zapisany w {split_file_path}")
