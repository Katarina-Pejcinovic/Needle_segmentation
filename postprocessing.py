import os
import numpy as np
from skimage import io, measure, morphology

#post-processing to remove little white ROIs

# Paths
input_directory = 'output_images_test'
output_directory = 'output_images_postprocessing'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Parameters
min_area = 90 

def process_image(file_path):

    image = io.imread(file_path, as_gray=True)

    # Apply connected components
    label_image = measure.label(image, connectivity=2)  
    
    # Create a boolean array where True indicates small regions
    small_objects = np.bincount(label_image.ravel()) < min_area
    small_objects[0] = False  # Ignore background

    # Remove small objects:
    cleaned_image = small_objects[label_image]

    # Convert to binary image (True becomes 1, False becomes 0)
    final_image = np.where(cleaned_image, 0, image).astype(np.uint8)

    return final_image

# Loop through all images in directory
for filename in os.listdir(input_directory):

    file_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    # Process the image
    processed_image = process_image(file_path)

    # Save the processed image
    io.imsave(output_path, processed_image)
