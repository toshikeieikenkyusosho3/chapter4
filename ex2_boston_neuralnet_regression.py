"""
例題2：ボストンハウジングデータ
複数の要因から住宅の価格を予測する
@author: t.imai
"""
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# 入力する部分
max_epochs = 1000                      # 訓練のステップの数
filename_train = 'ex2_boston_traindata.csv'    # 訓練用データのファイル名
filename_test = 'ex2_boston_testdata.csv'      # 確認用データのファイル名
explanatory_start_column = 2     # 説明変数の最初の列の列番号 2列目であれば2
explanatory_end_column =12      # 説明変数の最後の列の列番号 14列目であれば14
outcome_column = 13          # 目的変数（教師データ）の列番号 15列目であれば15
# 入力はここまで

explanatory_variables = np.arange(explanatory_start_column-1, explanatory_end_column)    
outcome_variables = [outcome_column-1]               

'''
データの生成
'''
df1 = pd.read_csv(filename_train, skiprows=[0], header=None)
df2 = np.array(df1.values.tolist())
x_train = df2[:, explanatory_variables]
y_train = df2[:, outcome_variables]    

df3 = pd.read_csv(filename_test, skiprows=[0], header=None)
df4 = np.array(df3.values.tolist())
x_test = df4[:, explanatory_variables]
y_test = df4[:, outcome_variables] 

# 入力データを正規化
x_min = x_train.min(axis=0)
x_max = x_train.max(axis=0)
x_train = (x_train - x_min)/(x_max - x_min)
x_test = (x_test - x_min)/(x_max - x_min)

# モデルの構築
model = tf.keras.models.Sequential([
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(64, activation='relu'), 
  keras.layers.Dense(64, activation='relu'),     
  keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=max_epochs, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#pd.set_option('display.max_rows', 500)
print(hist)
#hist.tail()

test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test mae:', test_scores[1])


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
 
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.legend()
 
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train loss')
  plt.legend()
  plt.show()

plot_history(history)

# 予測値をcsvファイルに出力
val_y = model.predict(x_test)    
np.savetxt('prediction_boston.csv', val_y, delimiter=',')
print (val_y)    