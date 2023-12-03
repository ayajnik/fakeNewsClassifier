try:
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
except ImportError as e:
    print(e)

class LSTMrnn:
    def __init__(self,dataPath):
        self.dataPath = dataPath
        self.data_for_modeling = dataPrep(self.dataPath)
        self.sent_length = 20
        self.vocab = 5000

    def splitData(self):

        ind_var,dep_var = self.data_for_modeling.target_independent_vars()
        emb_doc = self.data_for_modeling.padding_seq()

        X_final=np.array(emb_doc)
        y_final=np.array(dep_var)
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

        return X_train, X_test, y_train, y_test

    def createModel(self):
        embedding_vector_features=40
        model=Sequential()
        model.add(Embedding(self.vocab,embedding_vector_features,input_length=self.sent_length))
        #model.add(LSTM(100))
        model.add(LSTM(50, activation='relu', input_shape=(3,2)))

        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        return model

    def trainLSTMModel(self):

        global trainX,testX,trainy,testy

        lstmModel = self.createModel()
        trainX,testX,trainy,testy = self.splitData()
        print('Model Training initiated')
        lstmModel.fit(trainX,trainy,validation_data=(testX,testy),epochs=5,batch_size=64)
        embedding_vector_features=40
        print("Adding dropout...")
        lstmModel=Sequential()
        lstmModel.add(Embedding(self.vocab,embedding_vector_features,input_length=self.sent_length))
        lstmModel.add(Dropout(0.3))
        lstmModel.add(LSTM(100))
        lstmModel.add(Dropout(0.3))
        lstmModel.add(Dense(1,activation='sigmoid'))
        lstmModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print("Dropout added! :)")

    def evaluation(self):
        lstmModel = self.createModel()
        y_pred=lstmModel.predict(testX)
        confusion_matrix(testy,y_pred)
        print("Accuracy of the classifier model is:",accuracy_score(testy,y_pred))






    





