import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Import ML packages
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'Keras version: {tf.keras.__version__}')

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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


def threshold_decision(input, threshold):
    output = numpy.empty(shape=(input.shape[0], 248), dtype="uint8")
    for i, x in np.ndenumerate(input):
        output[i[0], i[1]] = 0 if x < threshold else 1
    return output


csv_x_train_url = "train_X.csv"
csv_y_train_url = "train_Y.csv"
csv_x_test_url = "test_X.csv"

df_x_train = pd.read_csv(csv_x_train_url).drop("ChallengeID", axis=1)
df_y_train = pd.read_csv(csv_y_train_url).drop("ChallengeID", axis=1)

#df_x_test = pd.read_csv(csv_x_test_url)
df_x_test = pd.read_csv(csv_x_test_url)

x = df_x_train.to_numpy()
y = df_y_train.to_numpy()
x_test = df_x_test.to_numpy()[:, 1:]

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20)

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

model = Sequential()
model.add(keras.Input(shape=x.shape[1]))
model.add(Dense(300, activation="relu"))
model.add(Dense(280, activation="relu"))
model.add(Dense(260, activation="relu"))
model.add(Dense(y.shape[1], activation="sigmoid"))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=100, batch_size=128, verbose=2)

plot_loss_acc(history)
plt.show()

predicted_values_validation = model.predict(x_validation)
fscore = f1_score(y_validation, threshold_decision(predicted_values_validation, 0.5), average='weighted')
print(f"F1 Score : {fscore}")


res = model.predict(x_test)
decision_mat = threshold_decision(res, 0.5)
df_x_test = df_x_test.iloc[0:,:249]
df_x_test = df_x_test.astype(dtype='int32')
df_x_test.iloc[0:, 1:] = decision_mat

df_x_test.to_csv("y_test.csv", index=False)
