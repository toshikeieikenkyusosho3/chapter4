"""
例題1：線形回帰モデル
施設の築後年数から保守コストを予想する
@author: t.imai
"""
%matplotlib inline

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
 
from tensorflow import keras
from tensorflow.keras import layers

# この例題では、入力値を直接プラグラム文にかきこんでいる。普通は別ファイルから読み込む。
x_train = np.array([[33], [23], [39], [45]])
y_train = np.array([[41000], [36000], [46000], [47000]])

model = tf.keras.models.Sequential([
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(64, activation='relu'), 
  keras.layers.Dense(64, activation='relu'),     
  keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=400, verbose=0)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  print(hist)  

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.legend()  
  plt.show()

plot_history(history)

x_test = [[30], [36]]
y_test = model.predict(x_test, verbose=2)
print ("\n Prediction using the developed model", \
       "\n Evaluation input:  ", x_test, \
       "\n Predicted values:  ", y_test)