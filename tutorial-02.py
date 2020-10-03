import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_label) = data.load_data(num_words=10000)

# Tratando as palavras, pois os dados de treino sao numeros
word_index = data.get_word_index()
word_index = { key: (v + 3) for key, v in word_index.items() }
word_index['<PAD>'] = 0
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(item, key) for (key, item) in word_index.items()])

# Ajustando os dados para terem o mesmo ou tamanho pareciso, nao entendi tudo
train_data = keras.preprocessing.sequences.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxLen=250)
test_data = keras.preprocessing.pad_sequences.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxLen=250)


def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])

model = keras.Sequential()
model.add(keras.layers.Embeddingg(10000, 16))
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))