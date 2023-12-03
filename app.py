from NLPmodels.rnnLSTM import LSTMClassifierRNN

train_Model = LSTMClassifierRNN.trainLSTMModel()
train_Model
eval_model = LSTMClassifierRNN.evaluation()
print(eval_model)