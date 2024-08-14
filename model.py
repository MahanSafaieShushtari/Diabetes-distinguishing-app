import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import time
import re
import tensorflow as tf
from tensorflow import keras as ks 
from sklearn.preprocessing import MinMaxScaler
firstdataset=pd.read_csv("/content/diabetes_dataset.csv")

maindataset=firstdataset.drop(["diabetes","year","location"],axis=1)
labels=firstdataset["diabetes"]
maindataset["gender"]=maindataset["gender"].apply(lambda x:1  if x=="Female" else 0)
dummies=pd.get_dummies(maindataset["smoking_history"])
maindataset= maindataset.drop(["smoking_history"],axis=1)

maindataset=pd.merge(maindataset,dummies,left_index=True,right_index=True)
maindataset=maindataset.drop(["No Info","not current","ever"],axis=1)
maindataset["never"]=maindataset["never"].apply(lambda x:1 if x==True else 0)
maindataset["former"]=maindataset["former"].apply(lambda x:1 if x==True else 0)
maindataset["current"]=maindataset["current"].apply(lambda x:1 if x==True else 0)
maindataset=np.asarray(maindataset)
labels=np.asarray(labels)
xtrain,xtest,ytrain,ytest=train_test_split(maindataset,labels,test_size=0.2)
scalar=MinMaxScaler()
maindataset=scalar.fit_transform(xtrain,(0,1))
maindataset=scalar.fit_transform(xtest,(0,1))

model=ks.Sequential()
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dense(units=1,activation="relu"))
model.compile(loss=tf.losses.MeanSquaredError,optimizer=tf.optimizers.Adam(),metrics=["accuracy"])
model.build(input_shape=(None,15))
model.summary()
hist=model.fit(xtrain,ytrain,batch_size=256,epochs=30,
               callbacks=ks.callbacks.EarlyStopping(monitor="val_loss",patience=10,min_delta=0.03),validation_data=(xtest,ytest))
plt.plot(hist.history["accuracy"],color="red")
plt.plot(hist.history["val_accuracy"],color="green")
plt.show()
