import tensorflow as tf
from model import EncoderCNN, DecoderRNN
from data import *
from preprocess_captions import clean_captions, get_tokenizer



caption_dir = "Flickr8k_text/Flickr8k.token.txt"
train_files_dir = "Flickr8k_text/Flickr_8k.trainImages.txt"
images_dir = "Flicker8k_Dataset/"
def evaluate(image_dir, max_length):
    image = tf.io.read_file(filename=image_dir)
    image = tf.io.decode_image(image)
    image_plot = tf.image.resize(image, (299, 299))
    image = tf.expand_dims(image_plot, 0)


    inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    inputs = inception.input
    outputs = inception.layers[-1].output

    feature_extractor = tf.keras.Model(inputs, outputs)

    extracted = feature_extractor(image)
    extracted = tf.reshape(extracted,
                     (extracted.shape[0], -1, extracted.shape[3]))
    # print(extracted.shape)

    encoder = EncoderCNN(embedding_dim=256)
    decoder = DecoderRNN(embedding_dim=256, units=512, vocab_size=7266)

    train_image_names = get_image_names(train_files_dir)
    caption_dict = get_captions_dict(caption_dir)
    caption_dict = valmap(lambda x: list(map(clean_captions, x)), caption_dict)
    train_caption_dict = filter_dict(caption_dict, train_image_names)

    tokenizer, vocab_size, max_caption_words = get_tokenizer(train_caption_dict)

    dec_input = tf.convert_to_tensor([[tokenizer.word_index['sos']]])
    hidden_state = decoder.reset_state(1)

    result = []

    for i in range(max_length):
        predictions, hidden_state, attention_weights = decoder(dec_input, extracted, hidden_state)
        predicted_token = tf.random.categorical(predictions, 1).numpy()[0][0]
        predicted_word = tokenizer.index_word[predicted_token]
        result.append(predicted_word)

        if predicted_word == '<EOS>':
            return result

        dec_input = tf.expand_dims([tokenizer.word_index[predicted_word]], 1)

    predicted_caption = " ".join(result)

    plt.imshow(image_plot / 255.0)
    plt.title(predicted_caption)
    plt.show()

    # print(predictions.shape, hidden_state.shape, attention_weights.shape)

evaluate("Flicker8k_Dataset/667626_18933d713e.jpg", 50)
