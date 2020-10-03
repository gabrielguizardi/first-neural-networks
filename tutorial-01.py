import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Diminui a escala dos numeros entre 0 a 1
train_images = train_images / 255
test_images = test_images / 255

# As imagens sao compostas de (28 x 28)px
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

# Define as sequencias das camadas
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),  # Input layer. Transforma a imagem em um array de 784 indexs.
  keras.layers.Dense(128, activation='relu'),  # Hidden layer. Define as ligacoes, como um grafo completo. E informando que sera uma activation function
  keras.layers.Dense(10, activation='softmax') # Output layer. Pega valores de cada neuronio, e define qual a probabilidade de ser determinada classe.
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(
  train_images,
  train_labels,
  epochs=5 # Quantas vezes o model ira ver os dados de treino, aleatoriamente ele pega as imagens e suas labels respectivas. A ordem dos dados de treino, influencia em como o modelo se ajusta
)

# Verifica a acertividade e o erro do modelo
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# Vai retornar um array com 10 indexes, na qual cada um representa uma classe, os dados presentes neles, sao a porcentagem de de certeza que o algoritimo tem
prediction = model.predict(test_images)

# mostra as imagens ja com a sua predicao descrita
for i in range(5):
  plt.grid(False)
  plt.imshow(test_images[i], cmap=plt.cm.binary)
  plt.xlabel('Actual: ' + class_names[test_labels[i]])
  plt.title('Prediction ' + class_names[np.argmax(prediction[i])])
  plt.show()
