import os
import shutil

# Ścieżki do pliku podziału i folderów
split_file_path = "../data/raw/ascii/train-test-split.txt"
processed_folder = "../data/processed/lines/"
train_folder = "../data/train/lines/"
test_folder = "../data/test/lines/"
val_folder = "../data/val/lines/"

# Tworzenie folderów docelowych
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Podział danych
with open(split_file_path, "r") as f:
    for line in f:
        parts = line.strip().split(" ")
        image_id = parts[0]
        set_type = parts[1]  # 'train', 'test', lub 'val'

        # Ścieżka źródłowa i docelowa
        src_path = os.path.join(processed_folder, f"{image_id}.png")
        if os.path.exists(src_path):
            if set_type == "train":
                shutil.move(src_path, train_folder)
            elif set_type == "test":
                shutil.move(src_path, test_folder)
            elif set_type == "val":
                shutil.move(src_path, val_folder)

print("Podzielono dane na zbiory treningowy, walidacyjny i testowy.")
