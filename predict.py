import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#讀取資料
df = pd.read_csv("netflix_titles_nov_2019.csv")

##################################決策樹模型##################################

#建立TfidfVectorizer和處理資料
TV = TfidfVectorizer(stop_words = "english",ngram_range=(1,2) ,min_df = 5)
X = TV.fit_transform(df["description"])
y = df["type"].copy()
y[y=="TV Show"]=0
y[y=="Movie"]=1
y_list = list(y)

#NMF
nmf = NMF(n_components=30, random_state=1)
X_nmf = nmf.fit_transform(X)

#使用DecisionTreeClassifier學習分類並預測分類(TfidfVectorizer)
DTclf = tree.DecisionTreeClassifier(random_state=1)
DTclf_scores = cross_val_score(DTclf,X,y_list,cv=10,scoring="accuracy")
print(DTclf_scores)
print(DTclf_scores.mean())

#使用RandomForestClassifier學習分類並預測分類(TfidfVectorizer)
RFclf = RandomForestClassifier(random_state=1, n_estimators=100)
RFclf_scores = cross_val_score(RFclf,X,y_list,cv=10,scoring="accuracy")
print(RFclf_scores)
print(RFclf_scores.mean())

#使用AdaBoostClassifier學習分類並預測分類(TfidfVectorizer)
AdaClf = AdaBoostClassifier(random_state=1)
AdaClf_scores = cross_val_score(AdaClf,X,y_list,cv=10,scoring="accuracy")
print(AdaClf_scores)
print(AdaClf_scores.mean())

#使用XGBClassifier學習分類並預測分類(TfidfVectorizer)
xgbClf = xgb.XGBClassifier(random_state=1)
xgbClf_scores = cross_val_score(xgbClf,X,y_list,cv=10,scoring="accuracy")
print(xgbClf_scores)
print(xgbClf_scores.mean())

#使用DecisionTreeClassifier學習分類並預測分類(NMF)
DTclfnmf = tree.DecisionTreeClassifier(random_state=1)
DTclfnmf_scores = cross_val_score(DTclfnmf,X_nmf,y_list,cv=10,scoring="accuracy")
print(DTclfnmf_scores)
print(DTclfnmf_scores.mean())

#使用RandomForestClassifier學習分類並預測分類(NMF)
RFclfnmf = RandomForestClassifier(random_state=1, n_estimators=100)
RFclfnmf_scores = cross_val_score(RFclfnmf,X_nmf,y_list,cv=10,scoring="accuracy")
print(RFclfnmf_scores)
print(RFclfnmf_scores.mean())

#使用AdaBoostClassifier學習分類並預測分類(NMF)
AdaClfnmf = AdaBoostClassifier(random_state=1)
AdaClfnmf_scores = cross_val_score(AdaClfnmf,X_nmf,y_list,cv=10,scoring="accuracy")
print(AdaClfnmf_scores)
print(AdaClfnmf_scores.mean())

#使用XGBClassifier學習分類並預測分類(NMF)
xgbClfnmf = xgb.XGBClassifier(random_state=1)
xgbClfnmf_scores = cross_val_score(xgbClfnmf,X_nmf,y_list,cv=10,scoring="accuracy")
print(xgbClfnmf_scores)
print(xgbClfnmf_scores.mean())

##################################神經網路模型##################################

#建立token
token = Tokenizer(num_words=4000)
token.fit_on_texts(df["description"])  
x_seq = token.texts_to_sequences(df["description"])
#padding
X_padding = sequence.pad_sequences(x_seq,maxlen=42)
#train test split
X_train,X_test,y_train,y_test= train_test_split(X_padding,y_list,test_size=0.2,random_state=1)

#glove pretrained
embeddings_index = {}
f = open("glove.42B.300d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype="float32")
    embeddings_index[word] = coefs
f.close()
word_index = token.word_index
embedding_matrix = np.zeros((4000,300))
for word, i in word_index.items():
    if(i>=4000):
        continue
    embedding_vector = embeddings_index.get(word)
    if(embedding_vector is not None):
        embedding_matrix[i] = embedding_vector

#RNN Dropout=0 
RNN_0 = Sequential()
RNN_0.add(Embedding(output_dim=32,input_dim=4000,input_length=42))
RNN_0.add(SimpleRNN(units=16))
RNN_0.add(Dense(units=256,activation="relu"))
RNN_0.add(Dense(units=1,activation="sigmoid"))
RNN_0.summary()
RNN_0.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
RNN_0_history = RNN_0.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
RNN_0_scores = RNN_0.evaluate(X_test,y_test,verbose=1)
print(RNN_0_scores[1])

#RNN Dropout=0.7
RNN_07 = Sequential()
RNN_07.add(Embedding(output_dim=32,input_dim=4000,input_length=42))
RNN_07.add(Dropout(0.7))
RNN_07.add(SimpleRNN(units=16))
RNN_07.add(Dense(units=256,activation="relu"))
RNN_07.add(Dropout(0.7))
RNN_07.add(Dense(units=1,activation="sigmoid"))
RNN_07.summary()
RNN_07.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
RNN_07_history = RNN_07.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
RNN_07_scores = RNN_07.evaluate(X_test,y_test,verbose=1)
print(RNN_07_scores[1])

#LSTM Dropout=0
LSTM_0 = Sequential()
LSTM_0.add(Embedding(output_dim=32,input_dim=4000,input_length=42))
LSTM_0.add(LSTM(32))
LSTM_0.add(Dense(units=256,activation="relu"))
LSTM_0.add(Dense(units=1,activation="sigmoid"))
LSTM_0.summary()
LSTM_0.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
LSTM_0_history = LSTM_0.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
LSTM_0_scores = LSTM_0.evaluate(X_test,y_test,verbose=1)
print(LSTM_0_scores[1])

#LSTM Dropout=0.7
LSTM_07 = Sequential()
LSTM_07.add(Embedding(output_dim=32,input_dim=4000,input_length=42))
LSTM_07.add(Dropout(0.7))
LSTM_07.add(LSTM(32))
LSTM_07.add(Dense(units=256,activation="relu"))
LSTM_07.add(Dropout(0.7))
LSTM_07.add(Dense(units=1,activation="sigmoid"))
LSTM_07.summary()
LSTM_07.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
LSTM_07_history = LSTM_07.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
LSTM_07_scores = LSTM_07.evaluate(X_test,y_test,verbose=1)
print(LSTM_07_scores[1])

#RNN Dropout=0(with glove pretrained embedding)
RNN_0_glove = Sequential()
RNN_0_glove.add(Embedding(4000,300,weights=[embedding_matrix],input_length=42,trainable=False))
RNN_0_glove.add(SimpleRNN(units=16))
RNN_0_glove.add(Dense(units=256,activation="relu"))
RNN_0_glove.add(Dense(units=1,activation="sigmoid"))
RNN_0_glove.summary()
RNN_0_glove.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
RNN_0_glove_history = RNN_0_glove.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
RNN_0_glove_scores = RNN_0_glove.evaluate(X_test,y_test,verbose=1)
print(RNN_0_glove_scores[1])

#RNN Dropout=0.7(with glove pretrained embedding)
RNN_07_glove = Sequential()
RNN_07_glove.add(Embedding(4000,300,weights=[embedding_matrix],input_length=42,trainable=False))
RNN_07_glove.add(Dropout(0.7))
RNN_07_glove.add(SimpleRNN(units=16))
RNN_07_glove.add(Dense(units=256,activation="relu"))
RNN_07_glove.add(Dropout(0.7))
RNN_07_glove.add(Dense(units=1,activation="sigmoid"))
RNN_07_glove.summary()
RNN_07_glove.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
RNN_07_glove_history = RNN_07_glove.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
RNN_07_glove_scores = RNN_07_glove.evaluate(X_test,y_test,verbose=1)
print(RNN_07_glove_scores[1])

#LSTM Dropout=0(with glove pretrained embedding)
LSTM_0_glove = Sequential()
LSTM_0_glove.add(Embedding(4000,300,weights=[embedding_matrix],input_length=42,trainable=False))
LSTM_0_glove.add(LSTM(32))
LSTM_0_glove.add(Dense(units=256,activation="relu"))
LSTM_0_glove.add(Dense(units=1,activation="sigmoid"))
LSTM_0_glove.summary()
LSTM_0_glove.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
LSTM_0_glove_history = LSTM_0_glove.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
LSTM_0_glove_scores = LSTM_0_glove.evaluate(X_test,y_test,verbose=1)
print(LSTM_0_glove_scores[1])

#LSTM Dropout=0.7(with glove pretrained embedding)
LSTM_07_glove = Sequential()
LSTM_07_glove.add(Embedding(4000,300,weights=[embedding_matrix],input_length=42,trainable=False))
LSTM_07_glove.add(Dropout(0.7))
LSTM_07_glove.add(LSTM(32))
LSTM_07_glove.add(Dense(units=256,activation="relu"))
LSTM_07_glove.add(Dropout(0.7))
LSTM_07_glove.add(Dense(units=1,activation="sigmoid"))
LSTM_07_glove.summary()
LSTM_07_glove.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
LSTM_07_glove_history = LSTM_07_glove.fit(X_train,y_train,epochs=100,batch_size=100,verbose=1,validation_split=0,shuffle=False)
LSTM_07_glove_scores = LSTM_07_glove.evaluate(X_test,y_test,verbose=1)
print(LSTM_07_glove_scores[1])
