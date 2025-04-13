import os
import shutil

input_folder = "../data/raw/lines/"
lines_file_path = "../data/raw/ascii/lines.txt"
filtered_folder = "../data/filtered/lines/"

os.makedirs(filtered_folder, exist_ok=True)

# Załaduj statusy z lines.txt
ok_files = []
with open(lines_file_path, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split(" ")
        if len(parts) >= 9 and parts[1] == "ok":
            ok_files.append(f"{parts[0]}.png")

# Przeszukiwanie folderów i przenoszenie plików "ok"
for root, _, files in os.walk(input_folder):
    for file_name in files:
        if file_name in ok_files:
            src_path = os.path.join(root, file_name)
            dest_path = os.path.join(filtered_folder, file_name)
            shutil.copy(src_path, dest_path)

print(f"Przeniesiono obrazy oznaczone jako 'ok' do folderu {filtered_folder}")
