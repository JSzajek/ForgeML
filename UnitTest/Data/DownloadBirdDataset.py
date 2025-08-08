from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm
import io
import base64
import json

# Load dataset from Hugging Face
dataset = load_dataset("yashikota/birds-525-species-image-classification", split="test")  # or "test", "validation"

# Output directory
output_root = "birds_525_images"
os.makedirs(output_root, exist_ok=True)

label_names = dataset.features["label"].names

# Create dict: index -> label name
label_dict = {i: name for i, name in enumerate(label_names)}

# Save to JSON
with open(output_root + "/label_map.json", "w") as f:
    json.dump(label_dict, f, indent=4)

print("Saved to label_map.json")
# Save images grouped by label
for item in tqdm(dataset, desc="Saving images"):
    label = item["label"]
    label_name = item["label_name"] if "label_name" in item else str(label)

    image_data = item["image"]

    # Get image object
    if isinstance(image_data, dict) and "bytes" in image_data:
        img = Image.open(io.BytesIO(image_data["bytes"]))
    else:
        img = image_data  # Already a PIL Image object

    # Create label folder
    label_dir = os.path.join(output_root, label_name.replace(" ", "_"))
    os.makedirs(label_dir, exist_ok=True)

    # Generate a filename
    file_name = f"{label_names[label]}.jpg"
    img.save(os.path.join(label_dir, file_name))
