import tensorflow as tf


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.weight1 = tf.keras.layers.Dense(units)
        self.weight2 = tf.keras.layers.Dense(units)
        self.weight3 = tf.keras.layers.Dense(1)

    def call(self, features, hidden_state):
        hidden_state = tf.expand_dims(hidden_state, 1)

        attention = tf.nn.tanh(self.weight1(features) + self.weight2(hidden_state))

        raw_scores = self.weight3(attention)

        attention_weights = tf.nn.softmax(raw_scores, axis=1)

        attention_output = attention_weights * features

        attention_output = tf.reduce_sum(attention_output, axis=1)

        return attention_output, attention_weights


class EncoderCNN(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(EncoderCNN, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class DecoderRNN(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(DecoderRNN, self).__init__()
        self.units = units
        self.attention = Attention(self.units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden_state):
        attention_output, attention_weights = self.attention(features, hidden_state)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(attention_output, 1), x], axis=-1)

        output, state = self.gru(x)

        x = self.fc1(output)

        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros([batch_size, self.units])

