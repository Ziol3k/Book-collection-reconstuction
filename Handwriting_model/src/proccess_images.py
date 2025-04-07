from PIL import Image
import os

# Ścieżki do folderów
input_folder = "../data/raw/lines/"
output_folder = "../data/processed/lines/"

# Tworzenie folderu docelowego
os.makedirs(output_folder, exist_ok=True)

# Odczytanie maksymalnych wymiarów
def find_max_dimensions(input_folder):
    max_width = 0
    max_height = 0
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".png"):
                img_path = os.path.join(root, filename)
                with Image.open(img_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
    return max_width, max_height

# Skalowanie i dodanie paddingu
def resize_and_pad_image(image, target_width, target_height, color=(255, 255, 255)):
    img = image.copy()
    img.thumbnail((target_width, target_height))
    new_image = Image.new("RGB", (target_width, target_height), color)
    offset = ((target_width - img.size[0]) // 2, (target_height - img.size[1]) // 2)
    new_image.paste(img, offset)
    return new_image

# Przetwarzanie wszystkich obrazów
def process_images(input_folder, output_folder, target_width, target_height):
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".png"):
                img_path = os.path.join(root, filename)
                with Image.open(img_path).convert("L") as image:
                    final_image = resize_and_pad_image(image, target_width, target_height)
                    output_path = os.path.join(output_folder, os.path.basename(filename))
                    final_image.save(output_path)

max_width, max_height = find_max_dimensions(input_folder)
print(f"Maksymalna szerokość: {max_width}, Maksymalna wysokość: {max_height}")

target_width = int(max_width * 1.1)
target_height = int(max_height * 1.1)

process_images(input_folder, output_folder, target_width, target_height)

print(f"Wszystkie obrazy zostały zapisane w {output_folder}")
