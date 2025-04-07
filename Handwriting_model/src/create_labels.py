import json

labels = {}

# Ścieżka do pliku lines.txt (dostosuj w razie potrzeby)
lines_file_path = "../data/raw/ascii/lines.txt"
output_path = "../data/processed/labels.json"

# Otwórz plik lines.txt
with open(lines_file_path, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split(" ")

        if len(parts) >= 9:
            image_id = parts[0]
            status = parts[1]
            text = " ".join(parts[8:]).replace("|", " ")

            if status == "ok":
                labels[f"{image_id}.png"] = text

# Zapisujemy wynik do pliku JSON
with open(output_path, "w") as f:
    json.dump(labels, f, indent=4)

print(f"Plik labels.json został zapisany w {output_path}")
