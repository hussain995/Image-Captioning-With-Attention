from data import load_image, get_image_names
from tqdm import tqdm
import numpy as np
import tensorflow as tf

images_dir = "Flicker8k_Dataset/"
train_files_dir = "Flickr8k_text/Flickr_8k.trainImages.txt"
train_image_names = get_image_names(train_files_dir)

inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
inputs = inception.input
outputs = inception.layers[-1].output

feature_extractor = tf.keras.Model(inputs, outputs)

training_image_paths = [images_dir + name + '.jpg' for name in train_image_names]

encode_train = sorted(set(training_image_paths))
# print("done first")
print(encode_train[:3])

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
# print("ran")
for img, paths in tqdm(image_dataset):
    feats = feature_extractor(img)
    feats = tf.reshape(feats,
                     (feats.shape[0], -1, feats.shape[3]))

    for feat, path in zip(feats, paths):
        feature_path = path.numpy().decode("utf-8").split(".")[0]
        print(feature_path)
        np.save(feature_path, feat.numpy())

# print("\n done")