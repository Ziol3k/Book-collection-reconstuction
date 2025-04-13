import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from PIL import Image
import json

# Dataset dla danych testowych
class IAMTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, labels):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, file_name)
        image = Image.open(image_path).convert("RGB")
        label = self.labels[file_name]
        return image, label

# Funkcja DataLoader
def collate_fn_test(batch):
    images, labels = zip(*batch)
    images = [processor(images=image, return_tensors="pt").pixel_values.squeeze(0) for image in images]
    images = torch.stack(images)
    labels = processor.tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True).input_ids
    return images, labels

# Ścieżki do folderów i danych testowych
test_folder = "data/test/lines/"  # Folder z obrazami do testów
labels_path = os.path.abspath("data/processed/labels.json")  # Plik z etykietami testowymi

# Załadowanie etykiet testowych
with open(labels_path, "r") as f:
    test_labels = json.load(f)

# Załaduj model i procesor
model_path = "../models/trocr_epoch_3"  # Ścieżka do wytrenowanego modelu (np. ostatnia epoka)
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

# Przygotowanie danych testowych
test_dataset = IAMTestDataset(test_folder, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_test)

# GPU lub CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Funkcja testowania modelu
def test_model(model, dataloader, processor, device, output_file="test_results2.txt"):
    model.eval()
    total_loss = 0  # Całkowity loss dla danych testowych
    num_batches = 0  # Licznik batchy

    with torch.no_grad():
        with open(output_file, "w") as result_file:
            result_file.write("Ground Truth vs Predictions\n")
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)

                # Przewidywania modelu
                outputs = model.generate(pixel_values=images, max_length=128)
                predictions = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ground_truths = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Obliczanie loss dla batcha
                outputs_with_loss = model(pixel_values=images, labels=labels)
                loss = outputs_with_loss.loss
                total_loss += loss.item()
                num_batches += 1

                # Zapisywanie wyników do pliku
                for gt, pred in zip(ground_truths, predictions):
                    result_file.write(f"GT: {gt} | Pred: {pred}\n")

                # Wyświetlanie postępów co 10 batchy
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"Test - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    # Obliczenie średniego Test Loss
    avg_test_loss = total_loss / num_batches
    print(f"Średni Test Loss: {avg_test_loss:.4f}")
    print(f"Wyniki zapisano w pliku: {output_file}")

# Uruchomienie testów
test_model(model, test_loader, processor, device)
