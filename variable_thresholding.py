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

from sklearn.model_selection import train_test_split

csv_x_train_url = "train_X.csv"
csv_y_train_url = "train_Y.csv"
csv_x_test_url = "test_X.csv"

df_x_train = pd.read_csv(csv_x_train_url).drop("ChallengeID", axis=1)
df_y_train = pd.read_csv(csv_y_train_url).drop("ChallengeID", axis=1)
df_x_test = pd.read_csv(csv_x_test_url)

x = df_x_train.to_numpy()
y = df_y_train.to_numpy()
x_test = df_x_test.to_numpy()

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25)

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")

def variable_threshold(input, output):
    threshold_tab = numpy.empty(248)
    threshold_tab[:] = 0.5

    for n in range(output.shape[0]):
        for col in range(output.shape[1]):
            if (input[n, col] > threshold_tab[col] and output[n, col] == 0) or \
                    (input[n, col] <= threshold_tab[col] and output[n, col] == 1):
                threshold_tab[col] = (threshold_tab[col]*(n+1) + input[n, col])/(n+2)
    return threshold_tab

def threshold_decision(input, v_threshold):
    input = input[:,:248]
    output = numpy.empty(shape=(input.shape[0], 248), dtype="uint8")
    for i, x in np.ndenumerate(input):
        output[i[0], i[1]] = 0 if x < v_threshold[i[1]] else 1
    return output


decision_mat = threshold_decision(x_test[:,1:], variable_threshold(x_train, y_train))

print(decision_mat)

print(df_x_test.head())

df_x_test = df_x_test.iloc[0:,:249]

df_x_test = df_x_test.astype(dtype='int32')
df_x_test.iloc[0:, 1:] = decision_mat
print(df_x_test.dtypes)

print(df_x_test.head())

df_x_test.to_csv("y_test.csv", index=False)
