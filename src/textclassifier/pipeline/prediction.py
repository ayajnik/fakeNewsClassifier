import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts/training/", "model.h5"))

        news = self.filename
        ps = PorterStemmer()

        corpus=[]
        
        for i in range(0,len(news)):
            review = re.sub('[^a-zA-Z]', ' ',str(news[i]))
            review=review.lower()
            review=review.split()
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review=' '.join(review)
            corpus.append(review)
        
        
        onehot_repr=[one_hot(words,5000)for words in corpus]
        embedded_docs = pad_sequences(onehot_repr,padding='pre',maxlen=20)
        result = np.argmax(model.predict(embedded_docs), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Fake news'
            return [{ "News" : prediction}]
        else:
            prediction = 'Original news'
            return [{ "News" : prediction}]