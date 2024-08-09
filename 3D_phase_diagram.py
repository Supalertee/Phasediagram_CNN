
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import plotly.graph_objects as go

def extract_integer(filename):
    return int(filename.split('.')[0])

model = tf.keras.models.load_model('model/2class.keras')
model.summary()


for elem in sorted(os.listdir('data/data_test3'), key = extract_integer):
   print(elem)

datatest = np.array([pd.read_csv(os.path.join("data/data_test3",elem)).to_numpy() for elem in sorted(os.listdir('data/data_test3'), key = extract_integer)])
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

np.savetxt("predict.csv", y_predict_new, delimiter=",")

y_value = np.array(y_predict_new).reshape((15,15,15))



X, Y, Z = np.mgrid[0:15, 0:15, 0:15]
values = y_value

# Create the volume plot
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0,
    isomax=1,

    opacity=0.05,  # Needs to be small to see through all surfaces
    surface=dict(count=100, fill=1),
    colorscale='RdBu'
))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='μ',  # Greek letter mu
            titlefont=dict(size=20, family='Arial', color='black'),
            tickvals=[0, 5, 10, 15],
            ticktext=['0.0', '0.25', '0.5', '0.75']
        ),
        yaxis=dict(
            title='η',  # Greek letter eta
            titlefont=dict(size=20, family='Arial', color='black'),
            tickvals=[0, 5, 10, 15],
            ticktext=['0.0', '0.25', '0.5', '0.75']
        ),
        zaxis=dict(
            title='T',  # Plain text
            titlefont=dict(size=20, family='Arial', color='black'),
            tickvals=[0, 5, 10, 15],
            ticktext=['0.1', '0.25', '0.5', '0.75'],
             range=[0, 7]
        )
    )
)
fig.show()


for i in range(0,100):
  plt.subplot(10,10,i+1),plt.imshow(datatest[i])
  plt.axis('off')
plt.show()

import plotly.graph_objects as go
import numpy as np

# Sample data generation (replace this with your actual data)

# Sample data
