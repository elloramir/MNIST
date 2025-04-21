import kagglehub
import os
import json
from PIL import Image
import numpy as np
import gzip

# Login to your Kaggle account first
# kagglehub.login()

path = kagglehub.dataset_download("mohamedgamal07/reduced-mnist")

def process_and_zip(data_folder, output_filename):
    final_data = {
        "width": None,
        "height": None,
        "data": []
    }

    print(f"Starting ripping the dataset: {data_folder}")

    for label in sorted(os.listdir(data_folder)):
        label_folder = os.path.join(data_folder, label)
        label_images = []

        print(f"Entering on label folder: {label}")

        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)

            # Loads the image in gray scale (single channel)
            img = Image.open(img_path).convert("L")

            # Set file header dimensions
            if final_data["width"] is None or final_data["height"] is None:
                final_data["width"], final_data["height"] = img.size

            img_array = np.array(img).flatten().tolist()
            label_images.append(img_array)

        # Add label images to the final dataset .json
        final_data["data"].append({
            "label": label,
            "images": label_images
        })

    # Compress the JSON data directly into a .gz file
    gz_filename = f"{output_filename}.gz"
    with gzip.open(gz_filename, 'wt', encoding='utf-8') as gz_file:
        json.dump(final_data, gz_file)

    print(f"Data processed and compressed into: {gz_filename}")


# Process both train and test folders and zip the output
train_path = os.path.join(path, "Reduced MNIST Data", "Reduced Trainging data")
test_path = os.path.join(path, "Reduced MNIST Data", "Reduced Testing data")

# Process the train and test data
process_and_zip(train_path, "train")
process_and_zip(test_path, "test")
