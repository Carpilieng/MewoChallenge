import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Import ML packages
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'Keras version: {tf.keras.__version__}')


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras # uniquement pour Input, faudrait, genre, import que ça
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


def plot_loss_acc(history):
    """Plot training and (optionally) validation loss and accuracy"""

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, '.--', label='Training loss')
    final_loss = loss[-1]
    title = 'Training loss: {:.4f}'.format(final_loss)
    plt.ylabel('Loss')
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'o-', label='Validation loss')
        final_val_loss = val_loss[-1]
        title += ', Validation loss: {:.4f}'.format(final_val_loss)
    plt.title(title)
    plt.legend()

    acc = history.history['accuracy']

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, '.--', label='Training acc')
    final_acc = acc[-1]
    title = 'Training accuracy: {:.2f}%'.format(final_acc * 100)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if 'val_accuracy' in history.history:
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, val_acc, 'o-', label='Validation acc')
        final_val_acc = val_acc[-1]
        title += ', Validation accuracy: {:.2f}%'.format(final_val_acc * 100)
    plt.title(title)
    plt.legend()

csv_x_train_url = "train_X.csv"
csv_y_train_url = "train_Y.csv"
#csv_x_test_url = "test_X.csv"

df_x_train = pd.read_csv(csv_x_train_url).drop("ChallengeID", axis=1)
df_y_train = pd.read_csv(csv_y_train_url).drop("ChallengeID", axis=1)

x = df_x_train.to_numpy()
y = df_y_train.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y_train = to_categorical(y_train)
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")

model = Sequential()
model.add(keras.Input(shape=x.shape[1]))
model.add(Dense(200, activation="sigmoid"))
model.add(Dense(180, activation="sigmoid"))
model.add(Dense(220, activation="sigmoid"))
model.add(Dense(y.shape[1], activation="relu"))

model.summary()

# opt = keras.optimizers.RMSprop(learning_rate=0.001, centered=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, batch_size=128)

plot_loss_acc(history)

val = x_train[0]
res = model.predict(x_train)
print(f"resultat: {res[0]}")
tab = np.copy(res[0])
for i in range(len(res[0])):
    tab[i] = 0 if res[0][i] < 0.15 else 1

print(f"rounded up: {tab}")

plt.show()
