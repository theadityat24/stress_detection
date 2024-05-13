import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Input, AveragePooling1D, Dropout
from keras.optimizers import Nadam
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler, Callback

from time import perf_counter
import functools

stresses = ['Gm', 'Drought', 'Nutrient_Deficiency', 'Fs', 'Salinity']
csv_path = r'..\combined.csv'
df = pd.read_csv(csv_path)
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df.drop(columns=['Fungal_infection'], inplace=True, errors='ignore')
df[stresses] = df[stresses].astype(bool)

spec_cols = [col for col in df.columns if col[0] == 'X']

df['Stressed'] = df[stresses].any(axis=1)

fuzzy_win = 5
blur_factor = 4

def cnn_reshape(x):
    return x.reshape((-1, x.shape[1], 1))

x = df[spec_cols].values
y = df['Stressed']

x /= x.max()

@functools.cache
def fuzzy_dx_init(shape, dtype=None):
    half_shape = list(shape)
    half_shape[0] //= 2
    half_shape = tuple(half_shape)
    return np.vstack((np.ones(half_shape) * -1/half_shape[0], np.ones(half_shape)/half_shape[0]))

model = Sequential([
    Input(shape=(x.shape[1],1)),
    Conv1D(1, fuzzy_win*2, trainable=False, kernel_initializer=fuzzy_dx_init),
    Conv1D(1, fuzzy_win*2, trainable=False, kernel_initializer=fuzzy_dx_init),
    Conv1D(1, fuzzy_win*2, trainable=False, kernel_initializer=fuzzy_dx_init),
    AveragePooling1D(blur_factor),
    Conv1D(20, 20, kernel_regularizer=L2(.001), name='conv1'),
    MaxPooling1D(2),
    Dropout(.02),
    Flatten(),
    Dense(40, kernel_regularizer=L2(.001)),
    Dropout(.02),
    Dense(30, kernel_regularizer=L2(.001)),
    Dropout(.02),
    Dense(10, kernel_regularizer=L2(.001)),
    Dropout(.02),
    Dense(1)
])

model.compile(optimizer=Nadam(1e-3, beta_1=.6, beta_2=.85), loss='binary_crossentropy', metrics=['accuracy'])

class FreezeCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch > 75:
            self.model.get_layer(name='conv1').trainable = False

def drop_lr(epochs, lr):
    if epochs < 50:
        return 1e-3
    elif epochs < 150:
        return 1e-4
    else:
        return 1e-5

def score(model, x, y, k=10, val_size=.2):
    results = np.empty((k,))
    train_i = np.random.rand(k,x.shape[0]) > val_size

    for i in range(k):
        start = perf_counter()

        x_train, x_val = x[train_i[i]], x[~train_i[i]]
        y_train, y_val = y[train_i[i]], y[~train_i[i]]

        weights = model.get_weights()

        model.fit(
            cnn_reshape(x_train),
            y_train,
            epochs=150,
            batch_size=100,
            callbacks = [LearningRateScheduler(drop_lr), FreezeCallback()],
            verbose=0,
        )

        y_pred = model.predict(cnn_reshape(x_val))

        model.set_weights(weights)

        accuracy = ((y_pred > y_pred.mean()).flatten() == y_val).mean()
        results[i] = accuracy

        print(f'{i}: {perf_counter() - start}')

    return results

results = score(model, x, y, k=5, val_size=.2)

print(results)
print(f'mean: {results.mean()}')
print(f'std: {results.std()}')

        