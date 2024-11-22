import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn

from xgboost import XGBClassifier

import tensorflow as tf

from keras.models import Model, Sequential
from keras import layers
from keras.optimizers import Adam
from keras.regularizers import L1, L2, Regularizer, L1L2
from keras import ops
from keras.callbacks import ReduceLROnPlateau
import keras

import imblearn

from time import perf_counter

np.random.seed(7)

PATH = '1.csv'

trait_cols = np.array([
    'Photo',
    'Ci', 'Cond', 'CTleaf', 'Trmmol', 'WUEi', 'WUEin', 'Fv_Fm', 'Fv_Fo',
    'PI', 'SLA', 'LWC', 'Suc', 'OP', 'OP100', 'RWC', 'WP', 'N', 'C',
    'Neoxanthin', 'Violaxanthin', 'Lutein', 'Zeaxanthin', 'Chl_b', 'Chl_a',
    'B_carotene', 'Glucose', 'Fructose', 'Sucrose', 'Sugars', 'Starch',
    'Ellagic', 'Gal', 'Rut', 'CTs'
])

stresses = ['Gm', 'Drought', 'Nutrient_Deficiency', 'Fs', 'Salinity']

class COReg(Regularizer):
    def __init__(self, lambda_1=1e-3, lambda_2=1e-3):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def call(self, x):
        return -1*self.lambda_1*ops.var(x) + L1(self.lambda_2)(x)

class CancelOutLayer(keras.layers.Layer):
    def __init__(self, lambda_1=1e-3, lambda_2=1e-3, **kwargs):
        super(CancelOutLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        weight_shape = (1, input_shape[-1])

        
        
        self.kernel = self.add_weight(
            name='kernel', 
            shape=weight_shape,
            initializer='uniform',
            trainable=True,
            regularizer=COReg
        )
        
        self.bias = self.add_weight(
            name='bias', 
            shape=weight_shape,
            initializer='zeros',
            trainable=True
        )
        
        self.built=True
    
    #operation:
    def call(self, inputs):
        return (inputs * self.kernel) + self.bias
    
    #output shape
    def compute_output_shape(self, input_shape):
        return input_shape

def get_acc(x_traits_train, x_traits_val, yi_train, yi_val, x_t_stress_train, x_t_stress_val):
    l_reg = 0

    individual_layers = []

    individual_layers.append(CancelOutLayer(lambda_1=0, lambda_2=0))

    for _ in range(6):
        individual_layers.append(layers.Dense(32, activation='relu', kernel_regularizer=L2(l_reg)))
        # individual_layers.append(layers.BatchNormalization())
        individual_layers.append(layers.Dropout(0.00))

    individual_layers.append(layers.Dense(yi_train.shape[1], activation='softmax'))

    individual_model = Sequential(individual_layers)

    individual_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    start = perf_counter()

    history = individual_model.fit(
        np.hstack((x_traits_train, x_t_stress_train)),
        yi_train,
        epochs=600,
        validation_data=(np.hstack((x_traits_val, x_t_stress_val)), yi_val),
        batch_size=3,
        verbose=0,
        # callbacks=[ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=30, min_lr=1e-5)]
    )

    print(perf_counter() - start)

    y_pred = individual_model.predict(np.hstack((x_traits_val, x_t_stress_val)))

    acc = (y_pred.argmax(axis=1) == yi_val.argmax(axis=1)).mean()

    temp_pred = y_pred.argmax(axis=1)
    temp_real = yi_val.argmax(axis=1)
    temp_pred[temp_pred == 3] = 0
    temp_real[temp_real == 3] = 0

    c_acc = (temp_pred == temp_real).mean()

    return acc, c_acc

def k_folds(PATH, k=5):      
    csv_path = PATH
    df = pd.read_csv(csv_path)
    df[stresses] = df[stresses].astype(bool)

    x_t = df['Time'].values.reshape((-1,1))

    stress_sel = df[stresses].any(axis=1)
    x_traits = df[stress_sel][trait_cols].values
    yi = df[stress_sel][stresses].values
    x_t_stress = x_t[stress_sel]

    x_traits = ((x_traits - x_traits.min(axis=0))/(x_traits.max(axis=0)-x_traits.min(axis=0)))

    sel_1 = yi.sum(axis=1).flatten() == 1
    x_traits = x_traits[sel_1]
    yi = yi[sel_1]
    x_t_stress = x_t_stress[sel_1]

    shuffle_sel = np.arange(x_traits.shape[0])
    np.random.shuffle(shuffle_sel)
    x_traits, yi, x_t_stress = x_traits[shuffle_sel], yi[shuffle_sel], x_t_stress[shuffle_sel]

    val_range = np.linspace(0, 1, x_traits.shape[0])
    accs = np.zeros((k,))
    c_accs = np.zeros((k,))

    for i in range(k):
        start = perf_counter()

        val_sel = (val_range >= i/k) & (val_range < (i+1)/k)

        x_traits_train, x_traits_val = x_traits[~val_sel], x_traits[val_sel]
        yi_train, yi_val = yi[~val_sel], yi[val_sel]
        x_t_stress_train, x_t_stress_val = x_t_stress[~val_sel], x_t_stress[val_sel]
        
        acc, c_acc = get_acc(x_traits_train, x_traits_val, yi_train, yi_val, x_t_stress_train, x_t_stress_val)
        accs[i] = acc
        c_accs[i] = c_acc

        print(f'PATH: {PATH}, i: {i}, time: {perf_counter() - start}, acc: {acc}')

    return accs, c_accs



print(k_folds(PATH))

df_dict = {'path':[], 'acc': [], 'c_acc': []}

paths = [
    r'1.csv',
    r'2.csv',
    r'3.csv',
    r'4.csv',
]

for path in paths:
    accs, c_accs = k_folds(path)

    print(f'acc: {accs.mean()}, c_acc: {c_accs.mean()}, acc_std: {accs.std()}, c_acc_std: {c_accs.std()}')

    for i in range(accs.shape[0]):
        df_dict['path'].append(path)
        df_dict['acc'].append(accs[i])
        df_dict['c_acc'].append(c_accs[i])

df = pd.DataFrame(df_dict)
df.to_csv('k_folds_results.csv')