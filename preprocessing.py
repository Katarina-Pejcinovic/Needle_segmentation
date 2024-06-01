'''preprocessing dictionary pkl file that contains training images and masks'''
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
mask_stack = masks_4d.astype(np.float32)

# Normalize the image stack to [0, 1]
image_stack /= 255.0
image_stack = image_stack.astype('float32')
mask_stack /= 255.0
image_stack = image_stack.astype('float32')

# Now, your image_stack is normalized along the entire stack
print(image_stack[0, :, :, :])
print(image_stack[0, :, :, :].shape)
print(image_stack.dtype)
print(masks_4d.dtype)

print(masks_4d[0, :, :, :])
print(mask_stack[0, :, :, :].shape)

print(mask_stack.dtype)
unique_elements, counts = np.unique(masks_4d, return_counts=True)
for element, count in zip(unique_elements, counts):
    print("before thresholding", f"{count} {element}s")

#save preprocessed images and masks
save_dataset(image_stack, "data/images_preprocessed.pkl")
save_dataset(mask_stack, "data/masks_preprocessed.pkl")


##############make binary labels for stratified k-fold############################

# Assuming `masks` is your numpy array of masks
labels = np.array([np.any(mask == 1) for mask in mask_stack]).astype(int)
print(labels)

save_dataset(labels, "data/cv_labels.pkl")
##############################preprocess test set############################

with open('data/test_images.pkl', 'rb') as f:
    test = pickle.load(f)
print("shape", test.shape)

print("test image shape", test[0, :, : ])
# #normalize the data 

# Convert the data type to float
image_stack = test.astype(np.float32)

# Normalize the image stack to [0, 1]
image_stack /= 255.0
image_stack = image_stack.astype('float32')

# Now is normalized along the entire stack
print(image_stack.shape)
print(image_stack[0, :, :, :])
print(image_stack.dtype)


save_dataset(image_stack, "data/test_processed.pkl")

#### make array for IDs


