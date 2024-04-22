import tensorflow as tf
from tensorflow import keras
import keras.api._v2.keras as keras
from keras import layers
import numpy as np
import pickle 
from make_sample_test import make_smaller_pkl, make_smaller
from keras import backend as K
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image


class BatchLossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BatchLossHistory, self).__init__()
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])

    def on_train_end(self, logs=None):
        plt.plot(self.batch_losses)
        plt.title('Loss per Batch')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.show()

def weighted_BCE_loss(y_true, y_pred, positive_weight=2):
   # y_true: (None,None,None,None)     y_pred: (None,512,512,1)
   y_pred = K.clip(y_pred, min_value=1e-12, max_value=1 - 1e-12)
   weights = K.ones_like(y_pred)  # (None,512,512,1)
   weights = tf.where(y_pred < 0.5, positive_weight * weights, weights)
   # weights[y_pred<0.5]=positive_weight
   out = keras.losses.binary_crossentropy(y_true, y_pred)  # (None,512,512)
   out = K.expand_dims(out, axis=-1) * weights  # (None,512,512,1)* (None,512,512,1)
   return K.mean(out)

#define U-net architecture 
def double_conv_block(x, n_filters):

   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)

   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x

def build_unet_model():
 # inputs
   inputs = layers.Input(shape=(512, 512, 1))

   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 32)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 64)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 128)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 128)

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 512)

   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 256)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 128)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 64)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 32)

   # outputs
   outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)  # Use 1 channel for binary mask

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model

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

unet_model = build_unet_model()
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=weighted_BCE_loss,
                  metrics=f1_score)

#make dataset tensorflow-compatible
with open('data/images_data.pkl', 'rb') as f:
    images = pickle.load(f)

with open('data/masks_data.pkl', 'rb') as f:
    masks = pickle.load(f)

# # make mini dataset for testing code
images = make_smaller(images)
masks = make_smaller(masks)

# Create TensorFlow dataset from images and masks
dataset = tf.data.Dataset.from_tensor_slices((images, masks))

# Define batch size
batch_size = 5

# Batch and shuffle the dataset
dataset = dataset.batch(batch_size).shuffle(buffer_size=1000)

total_num_samples = images.shape[0]
steps = total_num_samples//batch_size

print("steps per epoch", steps)

#train u-net
# Instantiate the custom callback
batch_loss_history = BatchLossHistory()
model_history = unet_model.fit(dataset, epochs = 1, steps_per_epoch = steps, callbacks=[batch_loss_history])

# Specify the path to save the model
# model_path = 'saved_model'
# unet_model.save(model_path)

#get mini test set 
# test = make_smaller_pkl('data/test_images.pkl')


with open('data/test_images.pkl', 'rb') as f:
    test = pickle.load(f)
print("test", test.shape)

masks_pred = unet_model.predict(test)


#save the predicted_masks
save_dir = 'outputs/'

#load in test_numbers.pkl
with open('data/test_numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)

def save_images_as_png(images_stack, numbers_array, output_directory):
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through the images stack and numbers array simultaneously
    for img_array, number in zip(images_stack, numbers_array):
        # Convert numpy array to PIL Image
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array.squeeze(axis=-1))  # Remove singleton channel axis if present

        # Save image as PNG with the desired filename format
        filename = os.path.join(output_directory, f"{number}.png")
        img = img.convert('L')
        img.save(filename)

        print(f"Saved image {number}.png")

save_images_as_png(masks_pred, numbers, 'output_images_real')

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


# #Display original images and predicted masks
# for i in range(test.shape[0]):
#     display_image_and_mask(test[i], masks_pred[i])