from data import *
import numpy as np
import tensorflow as tf
from train import train_step
from preprocess_captions import get_tokenizer
from model import Attention, EncoderCNN, DecoderRNN


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    caption_dir = "Flickr8k_text/Flickr8k.token.txt"
    train_files_dir = "Flickr8k_text/Flickr_8k.trainImages.txt"
    images_dir = "Flicker8k_Dataset/"

    BATCH_SIZE = 32
    BUFFER_SIZE = 1000

    train_image_names = get_image_names(train_files_dir)
    caption_dict = get_captions_dict(caption_dir)
    train_caption_dict = filter_dict(caption_dict, train_image_names)

    tokenizer, vocab_size, max_caption_words = get_tokenizer(train_caption_dict)

    images, captions = prep_data(train_caption_dict, tokenizer, max_caption_words, vocab_size, images_dir)

    NUM_STEPS = len(images) // BATCH_SIZE
    EPOCHS = 10

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_feats, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        running_loss = 0

        for i, inputs in enumerate(dataset):
            image_tensor, caption = inputs
            loss, total_loss = train_step(image_tensor, caption)
            # print("Loss, Total loss: ", loss, total_loss)
            running_loss += total_loss

            if i % 100 == 0:
                average_batch_loss = loss.numpy() / int(caption.shape[1])
                print(f'Epoch {epoch + 1} Batch {i} Loss {average_batch_loss:.4f}')
                # storing the epoch end loss value to plot later
        loss_plot.append(running_loss / num_steps)

        print(f'Epoch {epoch + 1} Loss {running_loss / num_steps:.6f}')
        # print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
