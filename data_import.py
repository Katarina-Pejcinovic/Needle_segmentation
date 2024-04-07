'''convert jpgs and masks to numpy arrays and save into dictionary '''
import os
import numpy as np
from PIL import Image
import pickle

def load_images_and_masks(image_dir, mask_dir):
    dataset2D = dict()
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    for image_file in image_files:
        
        if not image_file.endswith('.jpg'):
            continue

        # Extract ID number from filename
        image_number = os.path.splitext(image_file)[0]
        print(image_number)
        image_path = os.path.join(image_dir, image_file)
        mask_file_prefix = f"{image_number}_mask"
        matching_mask_files = [mask for mask in mask_files if mask.startswith(mask_file_prefix)]

        if len(matching_mask_files) != 1:
            continue  # Skip if there are no matching masks or multiple matching masks

        mask_path = os.path.join(mask_dir, matching_mask_files[0])

        # Load image and mask as grayscale
        image = np.array(Image.open(image_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        print(image.shape)

        dataset2D[image_number] = (image, mask)

    return dataset2D

def save_dataset(dataset, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")


image_directory = "/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/trainImages/trainImages"
mask_directory = "/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/trainMasks/trainMasks"
save_path = "dataset_dict.pkl"

# Load images and masks into dictionary
dataset2D = load_images_and_masks(image_directory, mask_directory)

# Save dataset
save_dataset(dataset2D, save_path)

    # # Example usage:
    # # Access image and mask for a specific number
    # number_to_access = "12345"  # Use the full ID number here
    # if number_to_access in dataset2D:
    #     image, mask = dataset2D[number_to_access]
    #     print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    # else:
    #     print(f"No data found for number {number_to_access}")



