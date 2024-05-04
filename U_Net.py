import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pickle 
from make_sample_test import make_smaller
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras.utils import plot_model
# print("Available devices:")
# print(tf.config.list_physical_devices())

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
        plt.savefig('figures/batch_loss.png')

def weighted_BCE(target, output, weights = [200, 1]):
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
   outputs = layers.Conv2D(1, (1,1), padding="same", activation = 'sigmoid')(u9) 

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model

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

unet_model = build_unet_model()
# plot_model(unet_model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss= 'binary_crossentropy',
                  metrics= [dice_coefficient])

#make dataset tensorflow-compatible
with open('data/images_data.pkl', 'rb') as f:
    images = pickle.load(f)

with open('data/masks_data.pkl', 'rb') as f:
    masks = pickle.load(f)

# make mini dataset for testing code
# images = make_smaller(images)
# masks = make_smaller(masks)
# Create TensorFlow dataset from images and masks
#dataset = tf.data.Dataset.from_tensor_slices((images, masks))

# Batches
batch_size = 30
epochs = 1
#dataset = dataset.batch(batch_size).repeat(count = epochs)
total_num_samples = images.shape[0]
steps = total_num_samples//batch_size

print("steps per epoch", steps)

#train u-net
batch_loss_history = BatchLossHistory()
# model_history = unet_model.fit(dataset, epochs = epochs, steps_per_epoch = steps, callbacks=[batch_loss_history])

#get mini test set 
#test = make_smaller_pkl('data/test_processed.pkl')

#######################functions for predictions and validations#######################
def apply_threshold(predictions, threshold=0.5):
    """Apply a binary threshold to segmentation predictions."""
    unique_elements, counts = np.unique(predictions, return_counts=True)
    #for element, count in zip(unique_elements, counts):
        #print("before thresholding", f"{count} {element}s")
    
    # Threshold the predictions
    predictions_thresholded = np.where(predictions < threshold, 0, 1)
    unique_elements, counts = np.unique(predictions_thresholded, return_counts=True)

    # Display the unique elements with their counts
    #for element, count in zip(unique_elements, counts):
        #print("After thresholding", f"{count} {element}s")

    return predictions_thresholded

def save_images_as_png(images_stack, numbers_array, output_directory):
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)
    print("images stack", images_stack.shape)

    # Iterate through the images stack and numbers array simultaneously
    for img_array, number in zip(images_stack, numbers_array):
        # Convert numpy array to PIL Image
        img_array = (img_array * 255).astype(np.uint8)
        #print("img_array", img_array.shape)
        img = Image.fromarray(img_array.squeeze(axis=-1))  # Remove singleton channel axis if present
        #unique_elements, counts = np.unique(img, return_counts=True)
        # Display the unique elements with their counts
        # for element, count in zip(unique_elements, counts):
        #     print("saving", f"{count} {element}s")
        
        filename = os.path.join(output_directory, f"{number}_mask.png")
        img = img.convert('L')
        img.save(filename)

        #print(f"Saved image {number}.png")

def predict_in_batches(data, numbers, batch_size=30):
    num_samples = data.shape[0]
    batch_numbers = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        #make predictions 
        batch_predictions = unet_model.predict(data[start:end])
        print("batch_predictions", batch_predictions.shape)
        batch_numbers = numbers[start:end]
        print("batch_numbers", len(batch_numbers), '\n', batch_numbers)
        predictions_thresh = apply_threshold(batch_predictions)
        print("predictions thresh", predictions_thresh.shape)
        #save_images_as_png(predictions_thresh, batch_numbers, 'output_images')

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
###################### Stratified Cross Validation##################
#load in labels for stratifying 
with open('data/strat_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

#numbers for saving images 
with open('data/test_numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)
print("numbers", numbers.shape)

# Number of splits
n_splits = 3

# Create the StratifiedKFold object
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
print("starting cross validation")
metrics_summary = pd.DataFrame(columns=['Fold', 'Dice Coefficient'])

fold_no = 1
for train_index, val_index in skf.split(images, labels):
    train_dataset = tf.data.Dataset.from_tensor_slices((images[train_index], masks[train_index]))
    train_dataset = train_dataset.batch(batch_size).repeat(count = epochs)
    val_dataset = tf.data.Dataset.from_tensor_slices((images[val_index], masks[val_index]))
    val_dataset = val_dataset.batch(batch_size)  # Assuming batch_size is defined

    # for images, masks in dataset.take(1):
    #     print("Batched images shape:", images.shape)  # e.g., (16, 512, 512, 1)
    #     print("Batched masks shape:", masks.shape)    # e.g., (16, 512, 512, 1)
    
    # train model
    print(f"Training on fold {fold_no}")
    unet_model = build_unet_model()
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss= 'binary_crossentropy',
                  metrics= [dice_coefficient])
    steps = images[train_index].shape[0]//batch_size
    print("steps in fold", steps)
    model_history = unet_model.fit(train_dataset, epochs = epochs, steps_per_epoch = steps, callbacks=[batch_loss_history])
    
    #validate model
    print("validating model")
    #predict_in_batches(val_images, numbers)
    eval_metrics = unet_model.evaluate(val_dataset, verbose=0)
    # Create a DataFrame for the current fold's metrics
    current_fold_metrics = pd.DataFrame({
        'Fold': [fold_no],
        'Loss': [eval_metrics[0]],
        'Dice Coefficient': [eval_metrics[1]]
    })

    # Concatenate the current fold's metrics DataFrame with the main summary DataFrame
    metrics_summary = pd.concat([metrics_summary, current_fold_metrics], 
                                ignore_index=True)

    fold_no += 1

print(metrics_summary)

################### miscellaneous ###########################

with open('data/images_data.pkl', 'rb') as f:
    images = pickle.load(f)

# # # make mini dataset for testing code
train = make_smaller(images)

with open('data/test_numbers.pkl', 'rb') as f:
    numbers = pickle.load(f)
print("numbers", numbers.shape)

with open('data/test_processed.pkl', 'rb') as f:
    test = pickle.load(f)

print("test", test.shape)
print("test sample", test[0, :, :, 0])

#predict_in_batches(images, numbers)



# predictions = unet_model.predict(train)
# for i in range(train.shape[0]):
#     display_image_and_mask(train[i], predictions[i])

# predictions = apply_threshold(predictions)

# #Display original images and predicted masks
# for i in range(train.shape[0]):
#     display_image_and_mask(train[i], predictions[i])


# #Display original images and predicted masks
# for i in range(test.shape[0]):
#     display_image_and_mask(test[i], masks_pred[i])