import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from src.textclassifier.entity.config_entity import TrainingConfig, EvaluationConfig
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
from src.textclassifier.utils.common import read_yaml, create_directories,save_json




class Evaluation:
    def __init__(self, config: EvaluationConfig):
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
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.padding_seq()
        trainX,testX,trainy,testy = self.splitData()
        self.score = self.model.evaluate(testX,testy)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            # if tracking_url_type_store != "file":

            # #     # Register the model
            # #     # There are other ways to use the Model Registry, which depends on the use case,
            # #     # please refer to the doc for more information:
            # #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #     mlflow.keras.log_model(self.model, "model")
            
            # else:
            #     mlflow.keras.save_model(self.model, "model")