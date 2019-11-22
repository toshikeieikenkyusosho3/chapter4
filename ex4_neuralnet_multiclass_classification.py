"""
例題4：MNIST - 手書き文字認識（画像認識）
@author: t.imai
"""
%matplotlib inline

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

 
from tensorflow import keras
from tensorflow.keras import layers

# 入力する部分
max_epochs = 500                       # 訓練のステップの数
# 入力データのうち、訓練に使うデータの割合。0.8であれば、80%が訓練に、20％が確認に使われる。
ratio_for_training = 0.8      
filename_data = 'ex4_MNIST_data.csv'    # データのファイル名
explanatory_start_column = 1      # 説明変数の最初の列の列番号
explanatory_end_column = 784     # 説明変数の最後の列の列番号
outcome_start_column = 785      # 目的変数の最初の列番号
outcome_end_column = 794      # 目的変数の最後の列番号
# 入力はここまで

'''
データの読み込み
'''
explanatory_variables = np.arange(explanatory_start_column-1, 
                                  explanatory_end_column)    
outcome_variables = np.arange(outcome_start_column-1, outcome_end_column)

df = pd.read_csv(filename_data, skiprows=[0], header=None)
df1 = np.array(df.values.tolist())    

# 列でXとYを分ける。上で指定した列の番号で分割する。
x_train = df1[:, explanatory_variables]
y_train = df1[:, outcome_variables]
    
# 入力データを正規化
#x_min = x_train.min(axis=0)
#x_max = x_train.max(axis=0)
#x_train = (x_train - x_min)/(x_max - x_min)
x_train=x_train/255.0

# モデルの構築
model = tf.keras.models.Sequential([
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(64, activation='relu'), 
  keras.layers.Dense(64, activation='relu'),     
  keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=max_epochs, validation_split=ratio_for_training, verbose=0)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
 
  print(hist)

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label='Validation Error')
  plt.legend()
 
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train loss')
  plt.plot(hist['epoch'], hist['val_loss'],
           label='Validation loss')
  plt.legend()
  plt.show()

plot_history(history)