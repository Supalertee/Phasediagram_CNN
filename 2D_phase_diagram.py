import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('model/2class.keras')
model.summary()



def extract_integer(filename):
    return int(filename.split('.')[0])


for elem in sorted(os.listdir('data/data_2d'), key = extract_integer):
   print(elem)


datatest = np.array([pd.read_csv(os.path.join("data/data_2d",elem)).to_numpy() for elem in sorted(os.listdir('data/data_2d'), key = extract_integer)])
size, dimension = datatest.shape[0] , int(np.sqrt(datatest.shape[1]))
datatest = datatest.reshape((size,dimension,dimension))
datatest = tf.convert_to_tensor(datatest ,dtype = tf.float32)


y_predict = model.predict(datatest)

i = 0
y_predict_new = []
while i < len(y_predict):
  if abs(y_predict[i][0] - y_predict[i][1]) < 0.2:
     y_predict_new.append(0.5)
     i += 1
  else:
    y_predict_new.append(np.argmax(y_predict[i]))
    i += 1

y_2d = np.array(y_predict_new).reshape((31,31))

x = np.arange(0,0.4,0.01333)
y = np.arange(0,0.4,0.01333)

X,Y = np.meshgrid(x,y)

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 20,
        }
plt.contourf(X,Y,y_2d,cmap = "RdBu")
plt.title("Mott Gap Phase Diagram")
plt.xlabel("copling constant  "r'$(\eta$)')
plt.ylabel("chemical potential  " r'$(\mu$)')
plt.text(0.15, 0.2, "Gapped Phase", fontdict=font)
plt.text(0.07, 0.04, "Gapless Phase", fontdict=font)

plt.show()


