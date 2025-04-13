if __name__ == '__main__':
    import os
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torch.profiler import profile, ProfilerActivity
    import json

    # Ścieżki do folderów
    train_folder = "data/train/lines/"
    val_folder = "data/val/lines/"
    labels_path = os.path.abspath("data/processed/labels.json")

    # Załadowanie etykiet
    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Dataset
    class IAMDataset(Dataset):
        def __init__(self, image_folder, labels, step=1):
            self.image_folder = image_folder
            # Filtrowanie co `step`-tą próbkę
            self.image_files = [f for i, f in enumerate(os.listdir(image_folder)) if f.endswith(".png") and i % step == 0]
            self.labels = labels

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            file_name = self.image_files[idx]
            image_path = os.path.join(self.image_folder, file_name)
            image = Image.open(image_path).convert("RGB")
            label = self.labels[file_name]
            return image, label

    # DataLoader
    def collate_fn(batch):
        images, labels = zip(*batch)
        # Przetwarzanie obrazów
        images = [processor(images=image, return_tensors="pt").pixel_values.squeeze(0) for image in images]
        images = torch.stack(images)
        # Tokenizacja etykiet z paddingiem i przycięciem
        labels = processor.tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True).input_ids
        return images, labels

    # Tworzenie zestawów danych z filtrowaniem co n-ty przykład
    step = 8  # Możesz zmienić tę wartość, np. na 4 dla co czwartego zdjęcia
    train_dataset = IAMDataset(train_folder, labels, step=step)
    val_dataset = IAMDataset(val_folder, labels, step=step)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model i procesor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    # Ustawienie `decoder_start_token_id` oraz `pad_token_id`
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # GPU lub CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optymalizator i funkcja straty
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Funkcja treningowa z komunikatami o postępach
    def train_epoch(model, dataloader, optimizer, device, epoch):
        model.train()
        total_loss = 0

        # Użycie profilera
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Wyświetlanie postępów co 10 batchy
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"Epoch {epoch + 1} - Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        # Wyświetlanie szczegółowych wyników profilera w terminalu
        print(prof.key_averages().table(sort_by="cuda_time_total"))

        return total_loss / len(dataloader)

    # Funkcja walidacyjna z komunikatami o postępach
    def validate_epoch(model, dataloader, device, epoch):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Wyświetlanie postępów co 10 batchy
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"Validation - Epoch {epoch+1} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    # Trening modelu
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_epoch(model, val_loader, device, epoch)
        print(f"Epoch {epoch+1} Completed - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Zapis modelu
    output_dir = "../models/trocr/"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model zapisany w {output_dir}")
