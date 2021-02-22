'''
This file is made by Hisham Madcor
CSE student at EJUST
Email: hisham.madcor@ejust.edu.eg
student id: 120270026
'''
'''
Make sure to install
all the required lipraries:
*natsort
*tflearn
*keras API
*matplotlib.
'''
import os
import cv2
import tflearn
import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow import keras
from natsort import os_sorted
import matplotlib.pyplot as plt
from keras.models import load_model
from tflearn.layers.conv import conv_2d, max_pool_2d
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#put the direct path of the directory of the subjects containing the GEI.
Data_DIR = "/home/hisham/Sensors/Visiual_Sensor_Gait_Data(VSGD)/gaittest/subjects/"

IMG_SIZE_X = 65
# IMG_SIZE_X = 88
IMG_SIZE_Y = 128

def create_train_data(path):
    Data = []
    directories = os_sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
    for dir in directories:
        for folder in os_sorted(os.listdir(path+dir)):
            if folder.startswith("GEI"):
                if folder.endswith("m"):
                    for  __,__, files in os.walk(path+dir+"/"+folder):
                        files.sort()
                    for file in files:
                        label = 1
                        img = cv2.imread(path+dir+"/"+folder+"/"+file,cv2.IMREAD_GRAYSCALE)
                        (cnts, _) = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
                        for c in cnts:
                            x,y,w,h = cv2.boundingRect(c)
                            roi=img[y:y+h,x:x+w].copy()
                        roi = cv2.resize(roi, (IMG_SIZE_X,IMG_SIZE_Y))
                        Data.append([np.array(roi),np.array(label)])
                elif folder.endswith("f"):
                    for  __,__, files in os.walk(path+dir+"/"+folder):
                        files.sort()
                    # Take the GEI cut the extra black edges and fi the size of it the the IMG_SIZE vars.
                    for file in files:
                        label = 0
                        img = cv2.imread(path+dir+"/"+folder+"/"+file,cv2.IMREAD_GRAYSCALE)
                        (cnts, _) = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
                        for c in cnts:
                            x,y,w,h = cv2.boundingRect(c)
                            roi=img[y:y+h,x:x+w].copy()
                        roi = cv2.resize(roi, (IMG_SIZE_X,IMG_SIZE_Y))
                        Data.append([np.array(roi),np.array(label)])
    shuffle(Data)
    # np.save('gender_recognition_data.npy', Data)
    return Data

data = create_train_data(Data_DIR)

# data = np.load("train_data.npy",allow_pickle=True)
train_data = data[:-233]
valdation_data = data[-233:-133]
test_data = data[-133:]

X_val = np.array([i[0] for i in valdation_data]).reshape(-1,IMG_SIZE_Y,IMG_SIZE_X,1)
Y_val = np.array([i[1] for i in valdation_data])
# print(X_val.shape)
# print(Y_val.shape)

X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE_Y,IMG_SIZE_X,1)
Y = np.array([i[1] for i in train_data])
# print(X.shape)
# print(Y.shape)


test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE_Y,IMG_SIZE_X,1)
test_y = np.array([i[1] for i in test_data])
# print(test_x.shape)
# print(test_y.shape)


model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(IMG_SIZE_Y,IMG_SIZE_X,1)))
model.add(keras.layers.MaxPool2D(2,2))

model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 128 neurons and Rectified Linear Unit activation function
model.add(keras.layers.Dense(128,activation='relu'))

# Output layer with single neuron which gives 0 for male or 1 for female
#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#using image generator.
dataAugmentaion = ImageDataGenerator(rotation_range = 30, zoom_range = 0.20,
fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True,
width_shift_range = 0.1, height_shift_range = 0.1)

#make an early stooping for the model after 25 time according to the state of val_loss
stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="auto")

#make a model checkpoint to save only the best model according to the testing Accuracy.
check = ModelCheckpoint('gender_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


Model = model.fit_generator(dataAugmentaion.flow(X, Y,batch_size=20),
 validation_data = (test_x, test_y), steps_per_epoch = len(X) //5,
 epochs = 500, callbacks=[stop,check], verbose= 1)

#Model valdation and testing:

Load_model = load_model('gender_best_model.h5')

# The trainig and the testing accuraccy of best trained model
_, train_acc = Load_model.evaluate(X, Y, verbose=0)
_, test_acc = Load_model.evaluate(test_x, test_y, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Testing the best model on completely new segment of data.
score_gender = Load_model.evaluate(X_val, Y_val, verbose=0)
print("Model validation(%s): %.2f%%" % (Load_model.metrics_names[1], score_gender[1]*100))

# Plotting The training and testing Loss and acuraccy.
fig, axes = plt.subplots(2, 1, constrained_layout=True)
loss ,acc = axes
loss.plot(Model.history['loss'], label='Train loss')
loss.plot(Model.history['val_loss'], label='Test loss')
loss.legend()
acc.plot(Model.history['acc'], label='Train acc')
acc.plot(Model.history['val_acc'], label='Test acc')
acc.legend()
plt.show()
