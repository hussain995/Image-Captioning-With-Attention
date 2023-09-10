from data import *
import re

# caption_dir = "Flickr8k_text/Flickr8k.token.txt"
# train_files_dir = "Flickr8k_text/Flickr_8k.trainImages.txt"
# images_dir = "Flicker8k_Dataset"
#
# caption_dict = get_captions_dict(caption_dir)

def clean_captions(sentence):
    sent = re.sub(r"[^a-zA-Z0-9]+", " ", sentence).lower()
    word_list = ['SOS'] + sentence.split() + ['EOS']
    word_list = [word for word in word_list if (len(word) > 1) and (word.isalpha())]
    return " ".join(word_list)

def all_captions (data_dict):
    return ([caption for key, captions in data_dict.items() for caption in captions])

def max_caption_length(captions):
    return max(len(caption.split()) for caption in captions)


def get_tokenizer(data_dict):
    captions = all_captions(data_dict)
    max_len = max_caption_length(captions)
    # print("max len in get_tokenizer", max_len)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(captions)

    vocab_size = len(tokenizer.word_index) + 1

    return tokenizer, vocab_size, max_len


def pad_caption(text, max_length):
    caption = pad_sequences([text], maxlen=max_length, padding="post")[0]

    return text