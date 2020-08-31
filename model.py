import re
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,cross_val_score,train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



import joblib


sw=stopwords.words('english')

def eda():
    data=pd.read_csv('data/IMDB Dataset.csv')
    print(data.head())
    print(data.columns)
    print(data.shape)

    '''for i in range(len(data)):
        text=str(data['review'][i])
        text=nltk.word_tokenize(text)
        data['review'][i]=text

    print('-------Tokenization------')
    print(data['review'].head())'''

    sentiments=data['sentiment']

    lemmantizer=WordNetLemmatizer()

    corpus=[]
    for i in range(len(data)):
        msg=re.sub('[^a-zA-Z]',' ',data['review'][i])
        msg=msg.lower()
        msg=msg.split()
        msg=[lemmantizer.lemmatize(word) for word in msg if not word in sw]
        msg=' '.join(msg)
        corpus.append(msg)

    corpus_data=pd.DataFrame(corpus)
    clean_data=pd.concat([corpus_data,sentiments],axis=1)
    clean_data.columns=['Review','Sentiment']

    clean_data.to_csv('Data/clean_data.csv',index=False)



def eda1():

    data=pd.read_csv('data/IMDB Dataset.csv')
    clean_data=pd.read_csv('data/clean_data.csv')
    print(data.head())
    print(clean_data.head())

    pcount=clean_data[clean_data['Sentiment']=='positive']
    ncount=clean_data[clean_data['Sentiment']=='negative']

    p=(len(pcount))
    n=(len(ncount))

    print(p,n)


    sns.countplot(clean_data['Sentiment'],hue=clean_data['Sentiment'])
    plt.show()





def modeling():
    clean_data=pd.read_csv('data/clean_data.csv')
    corpus=clean_data['Review']
    Y=clean_data['Sentiment']

    LE=LabelEncoder()
    Y=LE.fit_transform(Y)

    print(Y[:10])

    print(type(corpus))

    tf_idf=TfidfVectorizer()
    X=tf_idf.fit_transform(corpus)

    print(X.shape,X[:10])

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=42
                                                   )
    rfc=GradientBoostingClassifier()
    nbays=MultinomialNB()

    rfc.fit(x_train,y_train)
    nbays.fit(x_train,y_train)

    predictions=rfc.predict(x_test)
    predictions1 = nbays.predict(x_test)



    print(confusion_matrix(y_test,predictions1))
    print(classification_report(y_test,predictions1))

    print(accuracy_score(y_test,predictions1))


    joblib.dump(rfc, 'Models/Sentiment-Model-gbc.pkl')
    #joblib.dump(tf_idf, 'Models/Vectorizer.pkl')








if __name__ == '__main__':
    #eda()
    #eda1()
    modeling()

