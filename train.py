from model import EncoderCNN, DecoderRNN, Attention
from data import loss_function, get_image_names, filter_dict, get_captions_dict
from preprocess_captions import get_tokenizer, clean_captions
from cytoolz import valmap
import tensorflow as tf

caption_dir = "Flickr8k_text/Flickr8k.token.txt"
train_files_dir = "Flickr8k_text/Flickr_8k.trainImages.txt"
images_dir = "Flicker8k_Dataset/"

train_image_names = get_image_names(train_files_dir)
caption_dict = get_captions_dict(caption_dir)
caption_dict = valmap(lambda x: list(map(clean_captions, x)), caption_dict)
train_caption_dict = filter_dict(caption_dict, train_image_names)

tokenizer, vocab_size, max_caption_words = get_tokenizer(train_caption_dict)


EMBEDDING_DIM = 256
UNITS = 512
VOCAB_SIZE = vocab_size


encoder = EncoderCNN(EMBEDDING_DIM)
decoder = DecoderRNN(EMBEDDING_DIM, UNITS, VOCAB_SIZE)
attention = Attention(UNITS)
optimizer = tf.keras.optimizers.Adam()


def train_step(image_tensor, caption):

    loss = 0
    start_tokens = tf.convert_to_tensor([tokenizer.word_index['sos']] * caption.shape[0])
    decoder_input = tf.expand_dims(start_tokens, 1)

    hidden_state = decoder.reset_state(batch_size=caption.shape[0])

    with tf.GradientTape() as tape:
        image_features = encoder(image_tensor)

        for i in range(1, caption.shape[1]):
            predictions, hidden, _ = decoder(decoder_input, image_features, hidden_state)

            loss += loss_function(caption[:, i], predictions)
            # print("loss in train.py", loss)
            dec_input = tf.expand_dims(caption[:, i], 1)

    total_loss = loss / caption.shape[1]

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


