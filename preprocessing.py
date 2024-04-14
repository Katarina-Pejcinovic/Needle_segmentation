'''preprocessing'''
import pickle
from PIL import Image
import numpy as np
from data_import import save_dataset

with open('dataset_dict.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Extract images and masks into separate lists
images_list = []
masks_list = []

for key, (image, mask) in dataset.items():
    images_list.append(image)
    masks_list.append(mask)

# Convert lists to NumPy arrays
print(len(images_list))
images = np.array(images_list)
masks = np.array(masks_list)

print(images.shape)
print(masks.shape)

# Add another dimension to make it 4D
images_4d = np.expand_dims(images, axis=-1)
masks_4d = np.expand_dims(masks, axis=-1)

print("images", images_4d.shape)
print("masks", masks_4d.shape)

# #normalize the data 
import numpy as np

# Assuming your image stack is stored in the variable 'image_stack'
# image_stack.shape should be (505, 512, 512, 1)

# Convert the data type to float
image_stack = images_4d.astype(np.float32)

# Normalize the image stack to [0, 1]
image_stack /= 255.0
image_stack = image_stack.astype('float32')

# Now, your image_stack is normalized along the entire stack
print(image_stack[0, :, :, :])
print(image_stack[0, :, :, :].shape)
print(image_stack.dtype)
print(masks_4d.dtype)


save_dataset(image_stack, "data/images_data.pkl")
save_dataset(masks_4d, "data/masks_data.pkl")




