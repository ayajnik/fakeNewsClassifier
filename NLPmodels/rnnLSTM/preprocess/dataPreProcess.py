import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

class dataPrep:

    sen_length = 20
    vocab_size = 5000

    def __init__(self,data):
        self.data = data
        

    def read_data(self):
        df = pd.read_csv(self.data)
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
        onehot_repr=[one_hot(words,dataPrep.vocab_size)for words in stemmed]
        return onehot_repr

    def padding_seq(self):
        onehotted = self.oneHot()
        embedded_docs = pad_sequences(onehotted,padding='pre',maxlen=dataPrep.sen_length)
        return embedded_docs

    





    


    
