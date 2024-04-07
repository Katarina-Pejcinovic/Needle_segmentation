'''preprocessing'''
import pickle

with open('dataset_dict.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(dataset.keys())
number_to_access = "1164"
if number_to_access in dataset:
    image, mask = dataset[number_to_access]
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
else:
    print(f"No data found for number {number_to_access}")