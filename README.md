# Image Capitioning With Attention

This repository is my implementation of an image captioning model. I use an ecoder decoder model where I use the pretrained InceptionV3 model to extract image features and feed these
to the decoder RNN. Lastly, there is an Attention piece which is used by the decoder while predicting the next word to attend to different parts of the image. The dataset used is the
Flicker 8k dataset.
