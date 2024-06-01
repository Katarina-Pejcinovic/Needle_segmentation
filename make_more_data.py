import os
import numpy as np
from PIL import Image, ImageEnhance
import random
import pickle
import shutil



def aug_images_and_masks(image_dir, mask_dir, labels, output_image_dir, output_mask_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    for i, file_name in enumerate(image_files):
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name.replace('.jpg', '.png'))
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        if labels[i] == 1:
            # Random horizontal flip
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random contrast adjustment
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.5, 1.8)  # Randomly chosen factor for contrast adjustment
            image = enhancer.enhance(factor)
            
            # Save the processed image and mask to the output directories
            output_image_path = os.path.join(output_image_dir, file_name)
            output_mask_path = os.path.join(output_mask_dir, file_name.replace('.jpg', '.png'))
            
            image.save(output_image_path)
            mask.save(output_mask_path)
            print(f"Processed {file_name}: Label={labels[i]}, Flipped={labels[i] == 1}")
        else:
            print("did not save")

def contains_white_pixels(image):
    # Convert the image to grayscale
    grayscale_image = image.convert("L")
    # Convert the grayscale image to a NumPy array
    np_image = np.array(grayscale_image)
    # Check if any pixel is white (255)
    return np.any(np_image == 255)

def create_labels(image_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    labels = np.zeros(len(image_files), dtype=int)
    print("++++++++++++++++++++++++++")
    for i, file_name in enumerate(image_files):
        image_path = os.path.join(image_dir, file_name)
        image = Image.open(image_path)
        
        if contains_white_pixels(image):
            labels[i] = 1
        
        # Collect the file name and its corresponding label
        print(f"File: {file_name}, Label: {labels[i]}")

    return labels

def rename_and_copy_masks(mask_dir, new_dir):
    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    # List all mask files in the directory
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
    
    for mask_file in mask_files:
        # Extract the number from the filename
        number = mask_file.split('_')[0]
        # Create the new filename
        new_file_name = f"{number}.png"
        # Define the source and destination paths
        source_path = os.path.join(mask_dir, mask_file)
        destination_path = os.path.join(new_dir, new_file_name)
        # Copy and rename the file
        shutil.copy2(source_path, destination_path)
        print(f"Copied and renamed {source_path} to {destination_path}")

def delete_random_images_and_masks(image_dir, mask_dir, num_to_delete):
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Ensure the number of image files matches the number of mask files
    if len(image_files) != len(mask_files):
        print("The number of images does not match the number of masks.")
        return
    
    # Check if the number of files to delete is more than available files
    if num_to_delete > len(image_files):
        print("The number of images to delete exceeds the number of available images.")
        return
    
    # Randomly select files to delete
    files_to_delete = random.sample(image_files, num_to_delete)
    
    # Delete the selected files and corresponding masks
    for file_name in files_to_delete:
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name.replace('.jpg', '.png'))
        
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        if os.path.exists(mask_path):
            os.remove(mask_path)
            print(f"Deleted mask: {mask_path}")

def rename_files_with_111(image_dir, mask_dir):
    # Rename image files
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg'):
            base_name = file_name[:-4]  # Remove the .jpg extension
            new_file_name = base_name + '111.jpg'
            old_file_path = os.path.join(image_dir, file_name)
            new_file_path = os.path.join(image_dir, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")
    
    # Rename mask files
    for file_name in os.listdir(mask_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            base_name = file_name[:-4]  # Remove the .jpg or .png extension
            new_file_name = base_name + '111_mask' + file_name[-4:]  # Add 'a' before the extension
            old_file_path = os.path.join(mask_dir, file_name)
            new_file_path = os.path.join(mask_dir, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")

mask_directory = '/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/trainMasks/trainMasks'
new_directory_mask = '/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/trainMasks/newTrainMasks'
image_directory = '/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/trainImages/trainImages'  
output_image_directory = 'extra_needle_images'
output_mask_directory = 'extra_masks'

rename_and_copy_masks(mask_directory, new_directory_mask)

labels = create_labels(new_directory_mask)
print("label length, ", len(labels))

#Save the labels to a file
with open('data/needle_label_no_aug', 'wb') as f:
        pickle.dump(labels, f)

#make augmented images and corresponding masks
aug_images_and_masks(image_directory, new_directory_mask, labels, output_image_directory, output_mask_directory)
#create binary labels for needle/no needle



#randomly delete half of augmented images
num_to_delete = 176
delete_random_images_and_masks('extra_needle_images', 'extra_masks', num_to_delete)

new_image_directory = "extra_needle_images"
new_mask_directory = "extra_masks"

#rename
rename_files_with_111(new_image_directory, new_mask_directory)