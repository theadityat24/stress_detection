import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.regularizers import L2

import os
import csv
from time import perf_counter

import skopt

LOG_PATH = os.path.join('.', 'bayes_log.csv')\

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'noise_factor',
            'blur_factor',
            'kernel_size',
            'max_pool_n',
            'c_reg',
            'spatial_dropout_k',
            'n_cnn_blocks',
            'dropout_k',
            'n_20_blocks',
            'd_reg',
            'n_10_blocks',
            'accuracy',
            'time'
        ])
        writer.writeheader()



stresses = ['Gm', 'Drought', 'Nutrient_Deficiency', 'Fs', 'Salinity']

csv_path = os.path.join('..', '..', 'combined.csv')

df = pd.read_csv(csv_path)
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df.drop(columns=['Fungal_infection'], inplace=True, errors='ignore')
df[stresses] = df[stresses].astype(bool)
spec_cols = [col for col in df.columns if col[0] == 'X']
trait_cols = np.array(['Photo',
       'Ci', 'Cond', 'CTleaf', 'Trmmol', 'WUEi', 'WUEin', 'Fv_Fm', 'Fv_Fo',
       'PI', 'SLA', 'LWC', 'Suc', 'OP', 'OP100', 'RWC', 'WP', 'N', 'C',
       'Neoxanthin', 'Violaxanthin', 'Lutein', 'Zeaxanthin', 'Chl_b', 'Chl_a',
       'B_carotene', 'Glucose', 'Fructose', 'Sucrose', 'Sugars', 'Starch',
       'Ellagic', 'Gal', 'Rut', 'CTs'])

x_spec = df[spec_cols].values
yb = df[stresses].values.any(axis=1)

del df

def fuzzy_dx_init(shape, dtype=None):
    half_shape = list(shape)
    half_shape[0] //= 2
    half_shape = tuple(half_shape)
    return np.vstack((np.ones(half_shape) * -1/half_shape[0], np.ones(half_shape)/half_shape[0]))

def cnn_reshape(x):
    return x.reshape((-1, x.shape[1], 1))

x_spec = ((x_spec - x_spec.min(axis=0))/(x_spec.max(axis=0)-x_spec.min(axis=0)))
x_spec_train, x_spec_val, yb_train, yb_val = train_test_split(x_spec, yb, test_size=.2)

def ResBlock1D(x, kernel_size=10, max_pool_n=5, c_reg=.001):
    # padding has to be 'same' for add to work
    
    fx = layers.Conv1D(10, kernel_size, activation='relu', padding='same', kernel_regularizer=L2(c_reg))(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv1D(10, kernel_size, activation='relu', padding='same', kernel_regularizer=L2(c_reg))(fx)

    out = layers.Add()([x, fx])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    out = layers.MaxPooling1D(max_pool_n)(out)
    return out

dimensions = [
    skopt.space.Real(name='noise_factor', low=.0, high=.2),
    skopt.space.Integer(name='blur_factor', low=1, high=5),
    skopt.space.Integer(name='kernel_size', low=3, high=32),
    skopt.space.Integer(name='max_pool_n', low=1, high=10),
    skopt.space.Real(name='c_reg', low=.0, high=.1),
    skopt.space.Real(name='spatial_dropout_k', low=.0, high=.5),
    skopt.space.Integer(name='n_cnn_blocks', low=1, high=5),
    skopt.space.Real(name='dropout_k', low=.0, high=.5),
    skopt.space.Integer(name='n_20_blocks', low=1, high=5),
    skopt.space.Real(name='d_reg', low=.0, high=.1),
    skopt.space.Integer(name='n_10_blocks', low=1, high=5),
]

@skopt.utils.use_named_args(dimensions=dimensions)
def score(
    noise_factor,
    blur_factor,
    kernel_size,
    max_pool_n,
    c_reg,
    spatial_dropout_k,
    n_cnn_blocks,
    dropout_k,
    n_20_blocks,
    d_reg,
    n_10_blocks
):
    try:
        if x_spec_train.shape[1] < (max_pool_n **n_cnn_blocks * blur_factor):
            print(f'{max_pool_n} {n_cnn_blocks}')
            return 0
        start = perf_counter()
        fuzzy_win = 5

        cnn_model_layers = [
            layers.GaussianNoise(noise_factor),
            layers.Conv1D(1, fuzzy_win*2, trainable=False, kernel_initializer=fuzzy_dx_init),
            layers.AveragePooling1D((blur_factor,)),
        ] + [
            lambda x: ResBlock1D(x, kernel_size=(kernel_size,), max_pool_n=(max_pool_n,), c_reg=c_reg) for _ in range(n_cnn_blocks)
        ] + [
            layers.SpatialDropout1D(spatial_dropout_k) for _ in range(n_cnn_blocks)
        ] + [
            layers.Flatten()
        ] + [
            layers.Dense(20, activation='relu', kernel_regularizer=L2(d_reg)) for _ in range(n_20_blocks)
        ] + [
            layers.Dropout(dropout_k) for _ in range(n_20_blocks)
        ] + [
            layers.Dense(10, activation='relu', kernel_regularizer=L2(d_reg)) for _ in range(n_10_blocks)
        ] + [
            layers.Dropout(dropout_k) for _ in range(n_10_blocks)
        ] + [
            layers.Dense(1, activation='sigmoid')
        ]

        cnn_model_inputs = layers.Input(shape=(x_spec_train.shape[1],1))

        fx = cnn_model_inputs
        for layer in cnn_model_layers:
            fx = layer(fx)

        cnn_model_outputs = fx
        cnn_model = Model(inputs=cnn_model_inputs, outputs=cnn_model_outputs)

        cnn_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

        cnn_model.fit(
            cnn_reshape(x_spec_train),
            yb_train,
            epochs=500,
            validation_data=(cnn_reshape(x_spec_val), yb_val),
            batch_size=5,
            verbose=0
        )

        accuracy = ((cnn_model.predict(cnn_reshape(x_spec_val), verbose=0) > .5) == yb_val).mean()

        with open(LOG_PATH, 'a',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'noise_factor',
                'blur_factor',
                'kernel_size',
                'max_pool_n',
                'c_reg',
                'spatial_dropout_k',
                'n_cnn_blocks',
                'dropout_k',
                'n_20_blocks',
                'd_reg',
                'n_10_blocks',
                'accuracy',
                'time'
            ])

            writer.writerow({
                'noise_factor': noise_factor,
                'blur_factor': blur_factor,
                'kernel_size': kernel_size,
                'max_pool_n': max_pool_n,
                'c_reg': c_reg,
                'spatial_dropout_k':spatial_dropout_k,
                'n_cnn_blocks':n_cnn_blocks,
                'dropout_k':dropout_k,
                'n_20_blocks':n_20_blocks,
                'd_reg':d_reg,
                'n_10_blocks':n_10_blocks,
                'accuracy': accuracy,
                'time': perf_counter() - start
            })

        return accuracy
    except:
        print('error')
        return 0

res = skopt.gp_minimize(
    score,
    dimensions,
    random_state=7,
    n_calls=500
)