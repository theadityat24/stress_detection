import pandas as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Multiply
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

csv_path = 'combined.csv'
df = pd.read_csv(csv_path)

