import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
import pickle 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from pandas.plotting import table 
from skimage import io, measure, morphology, color


####################### functions and classes for running model #####################

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
        #plt.show()
        plt.savefig('batch_loss.png')

def weighted_BCE(target, output, weights = [1, 190]):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce

######################### define U-net architecture ################################

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

def build_unet_base():
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

   return inputs, u9

def threshold_lambda(x, threshold):
    thresholded = tf.cast(x > threshold, tf.float32)
    #tf.print("Thresholded Output:", thresholded)
    return thresholded

def print_tensor(x, message= "output"):
    tf.print(message, x)
    return x 

def build_unet_training_model():
    inputs, last_layer_output = build_unet_base()
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation='sigmoid')(last_layer_output)
    printed_outputs = layers.Lambda(print_tensor, arguments={'message': "Conv2D Output:"})(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="U-Net-Training")
    return model

#inference model contains thresholding
def build_unet_inference_model(threshold):
    inputs, last_layer_output = build_unet_base()
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation='sigmoid')(last_layer_output)
    thresholded_output = layers.Lambda(threshold_lambda, arguments={'threshold': threshold})(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=thresholded_output, name="U-Net-Inference")
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

    # Compute the Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


################### run on whole dataset ###########################
with open('data/images_preprocessed.pkl', 'rb') as f:
    images = pickle.load(f)

with open('data/masks_preprocessed.pkl', 'rb') as f:
    masks = pickle.load(f)

# initialize variables
batch_size = 30
epochs = 30
threshold = 0.8

total_num_samples = images.shape[0]
steps = total_num_samples//batch_size
print("steps per epoch", steps)

batch_loss_history = BatchLossHistory()

#Create TensorFlow dataset from images and masks
dataset = tf.data.Dataset.from_tensor_slices((images, masks))
dataset = dataset.batch(batch_size).repeat(count = epochs)

with open('data/test_numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)
print("numbers", numbers.shape)

with open('data/test_processed.pkl', 'rb') as f:
    test = pickle.load(f)

#train model
unet_model = build_unet_training_model()
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss= weighted_BCE,
                metrics= [dice_coefficient, tf.keras.metrics.Recall()])

model_history = unet_model.fit(dataset, epochs = epochs, steps_per_epoch = steps, callbacks=[batch_loss_history])

# Save the entire model to a file
unet_model.save('final_model.h5')



