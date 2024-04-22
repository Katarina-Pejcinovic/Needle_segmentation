import matplotlib.pyplot as plt
import numpy as np
import pickle 
from make_sample_test import make_smaller_pkl, make_smaller
import matplotlib.pyplot as plt
from keras import backend as K
import os
import tensorflow as tf
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def weighted_BCE_loss(y_true, y_pred, positive_weight=5):
   # y_true: (None,None,None,None)     y_pred: (None,512,512,1)
   y_pred = K.clip(y_pred, min_value=1e-12, max_value=1 - 1e-12)
   weights = K.ones_like(y_pred)  # (None,512,512,1)
   weights = tf.where(y_pred < 0.5, positive_weight * weights, weights)
   # weights[y_pred<0.5]=positive_weight
   out = keras.losses.binary_crossentropy(y_true, y_pred)  # (None,512,512)
   out = K.expand_dims(out, axis=-1) * weights  # (None,512,512,1)* (None,512,512,1)
   return K.mean(out)



def f1_score(y_true, y_pred):
   
    y_true = K.flatten(y_true)
    y_pred = K.round(K.flatten(y_pred))
    
    tp = K.sum(y_true * y_pred)  # True Positives
    fp = K.sum((1 - y_true) * y_pred)  # False Positives
    fn = K.sum(y_true * (1 - y_pred))  # False Negatives
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

model_path = 'saved_model'
# Load the model
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score, 'weighted_BCE_loss': weighted_BCE_loss})
print("Model loaded successfully.")

#get mini test set 
test = make_smaller_pkl('data/test_images.pkl')
# with open('data/test_images.pkl', 'rb') as f:
#     test = pickle.load(f)
# print("test", test.shape)

masks_pred = loaded_model.predict(test)


#save the predicted_masks
save_dir = 'outputs/'


def display_image_and_mask(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')  # Display the first channel (grayscale)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')  # Display the first channel of mask
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

#Display original images and predicted masks
for i in range(test.shape[0]):
    display_image_and_mask(test[i], masks_pred[i])

#load in test_numbers.pkl
with open('data/test_numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)

def save_images_as_png(images_stack, numbers_array, output_directory):
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through the images stack and numbers array simultaneously
    for img_array, number in zip(images_stack, numbers_array):
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array.squeeze(axis=-1))  # Remove singleton channel axis if present

        # Save image as PNG with the desired filename format
        filename = os.path.join(output_directory, f"{number}.png")
        img = img.convert('L')
        img.save(filename)

        print(f"Saved image {number}.png")

save_images_as_png(masks_pred, numbers, 'output_images_real')


# Example usage
# Assuming 'images_stack' contains the stack of images and 'numbers_array' contains the numbers corresponding to each image
# 'output_directory' is the directory where you want to save the PNG files
# Replace these placeholders with your actual data and directory path
# save_images_as_png(images_stack, numbers_array, output_directory)



# # Loop through each mask in the stack
# for idx, mask in enumerate(masks_pred):
#     # Check if all values in the mask are either 0 or 1
#     is_binary_mask = np.all((mask == 0) | (mask == 1))

#     if is_binary_mask:
#         print(f"Mask {idx+1} contains only binary values (0 and 1).")
#     else:
#         print(f"Mask {idx+1} does not contain only binary values.")