'''make a small sample to do predictions on for testing '''

import os
import numpy as np
from PIL import Image
import pickle

def images_to_numpy(directory_path, output_file_images, output_file_numbers):
    # Get list of image file names in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]  # Use '.png' extension

    # Sort image files to ensure consistent order
    image_files.sort()

    # Initialize empty lists to store image arrays and corresponding numbers
    image_arrays = []
    numbers_list = []

    for img_file in image_files:
        # Load image using PIL
        img_path = os.path.join(directory_path, img_file)
        img = Image.open(img_path)

        # Convert image to grayscale and then to numpy array
        img_array = np.array(img)

        # Append image array to the list
        image_arrays.append(img_array)

        # Extract number from filename and append to numbers list
        number = int(os.path.splitext(img_file)[0])  # Extract number from filename without extension
        numbers_list.append(number)

    # Stack image arrays along a new dimension to create a 3D numpy array
    stacked_array = np.stack(image_arrays, axis=0)
    images_3d = np.expand_dims(stacked_array, axis=-1)

    # Convert numbers list to numpy array
    numbers_array = np.array(numbers_list)

    # Save the stacked array as a pickle file
    with open(output_file_images, 'wb') as f:
        pickle.dump(images_3d, f)

    # Save the numbers array as a separate pickle file
    with open(output_file_numbers, 'wb') as f:
        pickle.dump(numbers_array, f)

    print(f"Images converted and stacked. Images saved as {output_file_images}")
    print(f"Numbers array saved as {output_file_numbers}")

# Example usage

# # Example usage:
input_directory = "/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/testImages/testImages"
output_pickle_file = "test_images.pkl"
output_number_file = "test_numbers.pkl"

#images_to_numpy(input_directory, output_pickle_file, output_number_file)
# with open('test_images.pkl', 'rb') as f:
#     test_images = pickle.load(f)
# print(test_images.shape)


def make_smaller_pkl(input_pickle_file):

    with open(input_pickle_file, 'rb') as f:
        stacked_array = pickle.load(f)

    # Check the shape of the stacked array
    print("Shape of stacked array:", stacked_array.shape)

    # Get a subsample of 5 images from the stack
    subsample_indices = [0, 2, 4, 6, 8]  # Example indices, adjust as needed
    subsample_images = stacked_array[subsample_indices]

    # Check the shape of the subsample
    print("Shape of subsample:", subsample_images.shape)
    
    return subsample_images
    
def make_smaller(stacked_array):

    # Check the shape of the stacked array
    print("Shape of stacked array:", stacked_array.shape)

    # Get a subsample of 5 images from the stack
    subsample_indices = [0, 2, 4, 6, 8, 10, 12, 14,16, 18, 20, 22, 25, 27, 30]  # Example indices, adjust as needed
    subsample_images = stacked_array[subsample_indices]

    # Check the shape of the subsample
    print("Shape of subsample:", subsample_images.shape)
    
    return subsample_images
    