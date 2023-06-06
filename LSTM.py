#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[2]:


# Define input sequence as array
data = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])


# In[3]:


# Define output sequence as array
labels = np.array([[0.4], [0.5], [0.6], [0.7]])


# In[4]:


# Reshape data to be 3-dimensional
data = data.reshape((data.shape[0], data.shape[1], 1))


# In[5]:


# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))


# In[6]:


# Compile the model
model.compile(optimizer='adam', loss='mse')


# In[7]:


# Train the model
model.fit(data, labels, epochs=200, verbose=0)


# In[8]:


# Evaluate the model
test_data = np.array([[0.5, 0.6, 0.7], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9]])
test_labels = np.array([[0.8], [0.9], [1.0]])
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
loss = model.evaluate(test_data, test_labels, verbose=0)
print(f'Test loss: {loss:.4f}')
