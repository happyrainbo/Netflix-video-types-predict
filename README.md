# Netflix-video-types-predict
資料探勘研究與實務  

程式目的：利用Netflix的影片文字描述來分類Netflix上的影片類別(電視劇或電影)，觀察影片的文字描述與其影片類別是否有其關聯性，幾個分類方法是否能夠成功分類。  

### 使用的模型
決策樹模型
 * Decision Tree Classifier
 * Random Forest Classifier
 * Ada Boost Classifier
 * XGB Classifier

神經網路模型
 * RNN
 * LSTM

### 實驗方法
資料前處理：針對影片的文字敘述進行TfidVectorizer、Non-negative Matrix Factorization、Token處理作為不同模型的輸入。

決策樹模型：對於4種決策樹模型，分別使用TfidVectorizer或Non-negative Matrix Factorization作為模型的輸入，進行10-Fold Cross Validation。

神經網路模型：對於2種神經網路模型，使用Token作為模型的輸入，分別調整Dropout為0或0.7，以及是否使用glove的pre trained word vectors。

### 實驗結果
決策樹模型
 * 使用TfidfVectorizer作為模型輸入的預測準確度都比Non-Negative Matrix Factorization作為模型輸入的預測準確度好。
 * DecisionTreeClassifier這種單一決策樹的預測準確度較差，約65%的預測準確度。
 * RandomForestClassifier、AdaBoostClassifier、XGBClassifier這三種多棵決策樹的預測準確度都較好，有約70%的預測準確度。

神經網路模型
 * 使用Token加上glove的pre trained word vectors的預測準確率相比僅使用Token有些許提升。
 * 使用Dropout=0.7對比使用Dropout=0可以提高預測準確率，可見在Dropout=0的時候有overfitting的情況，而Dropout=0.7有效避免了overfitting的發生。。
 * 神經網路模型整體比決策樹模型要好，能夠達到約73%的預測準確度。
