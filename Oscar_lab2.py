#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow
from tensorflow.keras.datasets import fashion_mnist


# In[5]:


#a Get the data

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
unsupervised_oscar = {'images':train_x}


# In[6]:


supervised_oscar = {'images':test_x, 'labels':test_y }


# In[7]:


#b Data Pre-preprocessing

unsupervised_oscar['images'] = unsupervised_oscar['images']/255
supervised_oscar['images'] = supervised_oscar['images']/255


# In[8]:


from tensorflow.keras.utils import to_categorical
supervised_oscar['labels'] = to_categorical(supervised_oscar['labels'])


# In[9]:


print("Unsup Images data shape:", unsupervised_oscar['images'].shape)
print("Sup Images data shape:", supervised_oscar['images'].shape)
print("Sup Labels data shape:", supervised_oscar['labels'].shape)


# In[10]:


#c Data Preparation (Training, Validation, Testing)

from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(17)
unsupervised_train_oscar, unsupervised_val_oscar = train_test_split(unsupervised_oscar['images'], train_size=0.95, random_state=17 )


# In[11]:


discard_oscarx, superv_oscarx, discard_oscary, superv_oscary  = train_test_split(supervised_oscar['images'], supervised_oscar['labels'], train_size=0.7, random_state=17 )


# In[12]:


x_train_oscar, x_temp, y_train_oscar, y_temp = train_test_split(superv_oscarx, superv_oscary, train_size=0.6, random_state=17 )


# In[13]:


x_val_oscar, x_test_oscar, y_val_oscar, y_test_oscar = train_test_split(x_temp, y_temp, train_size=0.5, random_state=17 )


# In[14]:


print("Unsup train shape:", unsupervised_train_oscar.shape)
print("Unsup val shape:", unsupervised_val_oscar.shape)
print("Train data shape:", x_train_oscar.shape)
print("Val data shape:", x_val_oscar.shape)
print("Test data shape:", x_test_oscar.shape)
print("Train label shape:", y_train_oscar.shape)
print("Val label shape:", y_val_oscar.shape)
print("Test label shape:", y_test_oscar.shape)


# In[15]:


#d Build, Train, and Validate a baseline CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D

cnn_v1_model_oscar = Sequential()
cnn_v1_model_oscar.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(28, 28, 1)))
cnn_v1_model_oscar.add(Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
cnn_v1_model_oscar.add(Flatten())
cnn_v1_model_oscar.add(Dense(100, activation='relu'))
cnn_v1_model_oscar.add(Dense(10, activation='softmax'))


# In[16]:


cnn_v1_model_oscar.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[17]:


cnn_v1_model_oscar.summary()


# In[18]:


cnn_v1_history_oscar = cnn_v1_model_oscar.fit(x_train_oscar, y_train_oscar,
          batch_size=256,
          epochs=10,
          verbose=1,
          validation_data=(x_val_oscar, y_val_oscar))


# In[79]:


#e Test and analyze the baseline model

import matplotlib.pyplot as plt
loss_train = cnn_v1_history_oscar.history['accuracy']
loss_val = cnn_v1_history_oscar.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[20]:


test_loss, test_acc = cnn_v1_model_oscar.evaluate(x_test_oscar, y_test_oscar, verbose=2)
print("Test accuracy:", test_acc)


# In[80]:


val_loss, val_acc = cnn_v1_model_oscar.evaluate(x_val_oscar, y_val_oscar, verbose=2)
print("Validation accuracy:", val_acc)


# In[21]:


cnn_predictions_oscar = cnn_v1_model_oscar.predict(x_test_oscar)


# In[22]:


cnn_predictions_oscar


# In[23]:


y_test_oscar


# In[24]:


from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_test, y_pred, labels):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    classes = len(labels)
    confmat = confusion_matrix(y_test, y_pred)
    print("          Confusion Matrix ")
    print(confmat)
    print("\n                  Classification Report ")
    print(classification_report(y_test, y_pred, target_names=labels))
    con = np.zeros((classes, classes))
    for x in range(classes):
        for y in range(classes):
            con[x, y] = confmat[x, y] / np.sum(confmat[x, :])

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.0) # for label size
    df = sns.heatmap(con, annot = True, fmt ='.2', cmap = 'Oranges', xticklabels = labels , yticklabels= labels)
    
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] 
plot_confusion_matrix(y_test_oscar, cnn_predictions_oscar, labels) 


# In[25]:


#f Add random noise to unsupervised dataset

import tensorflow as tf
noise_factor = 0.2
tf.random.set_seed(17)

noise_train = tf.random.normal(unsupervised_train_oscar.shape, 0, 1)
noise_val = tf.random.normal(unsupervised_val_oscar.shape, 0, 1)

x_train_noisy_oscar = unsupervised_train_oscar + noise_factor + noise_train
x_val_noisy_oscar = unsupervised_val_oscar + noise_factor + noise_val


# In[26]:


x_train_noisy_oscar = tf.clip_by_value(x_train_noisy_oscar, clip_value_min=0, clip_value_max=1)
x_val_noisy_oscar = tf.clip_by_value(x_val_noisy_oscar, clip_value_min=0, clip_value_max=1)


# In[27]:


plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_val_noisy_oscar[i])        
plt.show()


# In[49]:


#g Build and pretrain Autoencoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2DTranspose

inputs_oscar = Input(shape=(28, 28, 1))

#encoder
e_oscar = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(inputs_oscar)
e_oscar = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(e_oscar)
           
#decoder
d_oscar = Conv2DTranspose(8, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(e_oscar)
d_oscar = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(d_oscar)
d_oscar = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(d_oscar)           


# In[38]:


autoencoder_oscar = Model(inputs=inputs_oscar, outputs=d_oscar, name='autoencoder_oscar')
# compile autoencoder model
autoencoder_oscar.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[39]:


autoencoder_oscar.summary()


# In[31]:


ae_history_oscar = autoencoder_oscar.fit(x_train_noisy_oscar, unsupervised_train_oscar,
          batch_size=256,
          epochs=10,
          verbose=1,
          validation_data=(x_val_noisy_oscar, unsupervised_val_oscar), shuffle=True)


# In[32]:


autoencoder_predictions_oscar = autoencoder_oscar.predict(unsupervised_val_oscar)


# In[33]:


plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(autoencoder_predictions_oscar[i])        
plt.show()


# In[69]:

#h Build and perform transfer learning on a CNN with the Autoencoder

cnn1 = autoencoder_oscar(inputs_oscar)
cnn2 = e_oscar
cnn3 = Flatten()(autoencoder_oscar.layers[-1].output)
cnn4 = Dense(100, activation='relu')(cnn3)
output = Dense(10, activation='softmax')(cnn4)

cnn_v2_oscar= Model(inputs=autoencoder_oscar.inputs, outputs=output)


# In[70]:


cnn_v2_oscar.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[71]:


cnn_v2_oscar.summary()


# In[72]:


cnn_v2_history_oscar= cnn_v2_oscar.fit(x_train_oscar, y_train_oscar,
          batch_size=256,
          epochs=10,
          verbose=1,
          validation_data=(x_val_oscar, y_val_oscar))


# In[74]:


#i Test and analyze the pretrained CNN model

loss_train1 = cnn_v2_history_oscar.history['accuracy']
loss_val1 = cnn_v2_history_oscar.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_train1, 'y', label='Training accuracy')
plt.plot(epochs, loss_val1, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[75]:


test_loss1, test_acc1 = cnn_v2_oscar.evaluate(x_test_oscar, y_test_oscar, verbose=2)
print("Test accuracy:", test_acc1)


# In[81]:


val_loss1, val_acc1 = cnn_v2_oscar.evaluate(x_val_oscar, y_val_oscar, verbose=2)
print("Validation accuracy:", val_acc1)


# In[76]:


cnn_predictions1_oscar = cnn_v2_oscar.predict(x_test_oscar)


# In[77]:


plot_confusion_matrix(y_test_oscar, cnn_predictions1_oscar, labels) 


# In[78]:


#j Compare the performance of the baseline CNN model to the pretrained model in your report

loss_valcnn = cnn_v1_history_oscar.history['val_accuracy']
loss_valpre = cnn_v2_history_oscar.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_valcnn, 'b', label='Validation accuracy Baseline')
plt.plot(epochs, loss_valpre, 'y', label='Validation accuracy Pretrained')
plt.title('Validation accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




