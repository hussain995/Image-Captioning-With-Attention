import os
import random
import numpy as np
import matplotlib.pyplot as plt
from cytoolz.dicttoolz import valmap
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess_captions import clean_captions
import re
# print(os.listdir(os.getcwd()))




def get_captions_dict(directory):
    with open(directory) as file:
        captions = file.readlines()

    stop_val = len(captions) - 1

    i = 0
    out = {}

    while i < stop_val:
        cur_list = captions[i: i + 5]
        filename = cur_list[0].split("#")[0].split(".")[0]
        for description in cur_list:
            out[filename] = list(map(lambda x: x.split("\t")[1].strip(), cur_list))
        i += 5
    return out

# caption_dict = get_captions_dict(caption_dir)
# print(caption_dict)

def filter_dict(captions_dict, image_names):
    out = {image_name: caption_list for image_name, caption_list in captions_dict.items() if image_name in image_names}
    return out

def get_image_names(directory):
    names = []

    with open(directory) as file:
        file_names = file.readlines()

    return list(set((map(lambda x: x.split(".")[0], file_names))))


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def plot_random_img(directory):
    img_name = random.choice(os.listdir(directory))
    description = random.choice(caption_dict[img_name])
    path = directory + "/" + img_name
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img)
    plt.imshow(img)
    plt.title(description)

    return img

def prep_data(data_dict, tokenizer, max_length, vocab_size, images_dir):
    X = []
    y = []

    caption_dict = valmap(lambda x: list(map(clean_captions, x)), data_dict)

    for image, caption_list in caption_dict.items():
        file_name = images_dir + image + ".jpg"

        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            #             print(seq)
            padded_seq = pad_sequences([seq], maxlen=max_length, padding='post')[0]
            #             print(padded_seq)
            X.append(file_name)
            y.append(padded_seq)

    return X, y

def load_feats(image_name, caption):
    img_tensor = np.load(image_name.decode("utf-8").split(".")[0] + ".npy")
    return img_tensor, caption

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# img = plot_random_img(images_dir)