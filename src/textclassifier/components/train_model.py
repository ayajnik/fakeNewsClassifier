import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from src.textclassifier.entity.config_entity import TrainingConfig
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout




nltk.download('stopwords')

class Training:
    def __init__(self,config: TrainingConfig):
        self.config = config
        

    def read_data(self):
        df = pd.read_csv(self.config.training_data)
        return df

    def target_independent_vars(self):
        df_inputData = self.read_data()
        X = df_inputData.drop('label',axis=1)
        y = df_inputData['label']
        return X,y

    def corpusData(self):

        global messages
        independent_vars,dependent_vars = self.target_independent_vars()

        messages = independent_vars.copy()

        messages.reset_index(inplace=True)

        nltk.download('stopwords')
        return messages

    def stemming(self):

        ps = PorterStemmer()

        preparedData = self.corpusData()
        corpus=[]
        
        for i in range(0,len(preparedData)):
            review = re.sub('[^a-zA-Z]', ' ',str(preparedData['title'][i]))
            review=review.lower()
            review=review.split()
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review=' '.join(review)
            corpus.append(review)

        return corpus

    def oneHot(self):
        stemmed = self.stemming()
        onehot_repr=[one_hot(words,self.config.params_is_vocab)for words in stemmed]
        return onehot_repr

    def padding_seq(self):
        onehotted = self.oneHot()
        embedded_docs = pad_sequences(onehotted,padding='pre',maxlen=self.config.params_is_sen_length)
        return embedded_docs
    
    def get_base_model(self):
        model = tf.keras.models.load_model(
            self.config.base_model_path
        )
        return model

    def splitData(self):

        ind_var,dep_var = self.target_independent_vars()
        emb_doc = self.padding_seq()

        X_final=np.array(emb_doc)
        y_final=np.array(dep_var)
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def trainLSTMModel(self):

        global trainX,testX,trainy,testy

        lstmModel = self.get_base_model()
        trainX,testX,trainy,testy = self.splitData()
        print('Model Training initiated')
        lstmModel.fit(trainX,trainy,validation_data=(testX,testy),epochs=self.config.params_is_epochs,batch_size=self.config.params_is_batch_size)
        embedding_vector_features=40
        print("Adding dropout...")
        lstmModel=Sequential()
        lstmModel.add(Embedding(self.config.params_is_vocab,embedding_vector_features,input_length=self.config.params_is_sent_length))
        lstmModel.add(Dropout(0.3))
        lstmModel.add(LSTM(100))
        lstmModel.add(Dropout(0.3))
        lstmModel.add(Dense(1,activation='sigmoid'))
        lstmModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print("Dropout added! :)")

        self.save_model(
            path=self.config.trained_model_path,
            model=lstmModel
        )
    
    

    

    