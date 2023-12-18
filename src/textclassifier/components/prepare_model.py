try:
    from pathlib import Path
    from src.textclassifier.entity.config_entity import PrepareModelConfig
    import tensorflow as tf
    import pandas as pd
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.text import one_hot
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense
    from NLPmodels.rnnLSTM.preprocess import dataPrep
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Dropout
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    import numpy as np
    import os
    import urllib.request as request
    from zipfile import ZipFile
except ImportError as e:
    print(e)

class PrepareBaseModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def createModel(self):
        embedding_vector_features=40
        model=Sequential()
        model.add(Embedding(self.config.params_vocab,embedding_vector_features,input_length=self.config.params_sent_length))
        #model.add(LSTM(100))
        model.add(LSTM(50, activation='relu'))

        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        return model

    def trainLSTMModel(self,trainX,trainy,testX,testy):

        #global trainX,testX,trainy,testy

        lstmModel = self.createModel()
        #trainX,testX,trainy,testy = self.splitData()
        print('Model Training initiated')
        lstmModel.fit(trainX,trainy,validation_data=(testX,testy),epochs=self.config.epochs,batch_size=self.config.batch_size)
        embedding_vector_features=40
        print("Adding dropout...")
        lstmModel=Sequential()
        lstmModel.add(Embedding(self.config.vocab,embedding_vector_features,input_length=self.config.sent_length))
        lstmModel.add(Dropout(0.3))
        lstmModel.add(LSTM(100))
        lstmModel.add(Dropout(0.3))
        lstmModel.add(Dense(1,activation='sigmoid'))
        lstmModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print("Dropout added! :)")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    