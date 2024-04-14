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
                  loss="binary_crossentropy",
                  metrics=f1_score)

#make dataset tensorflow-compatible
with open('data/images_data.pkl', 'rb') as f:
    images = pickle.load(f)

with open('data/masks_data.pkl', 'rb') as f:
    masks = pickle.load(f)

#make mini dataset for testing code
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
model_history = unet_model.fit(dataset,
                              epochs = 3,
                              steps_per_epoch = steps)

# Specify the path to save the model
model_path = 'saved_model'
# Save the model
print("heres")
unet_model.save(model_path)

# Plot training loss
plt.plot(model_history.history['loss'], label='Training Loss')
print(model_history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

