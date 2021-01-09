import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os.path
from os import path

def preprocess_data():
    df = pd.read_csv('emails.csv')
    df.drop_duplicates(inplace=True)
    nltk.download('stopwords')

    messagesBow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

    return messagesBow, df


def split_data(messagesBow, df):
    X_train, X_test, y_train, y_test = train_test_split(messagesBow, df['spam'], test_size=0.20, random_state=0)
    print(X_test)
    return X_train, X_test, y_train, y_test


def make_model(messagesBow, df):

    X_train, X_test, y_train, y_test = split_data(messagesBow, df)

    print(X_test)
    print(messagesBow.shape)

    classifier = MultinomialNB().fit(X_train, y_train)

    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    print(classifier.predict(X_train))

    print(y_train.values)

    pred = classifier.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, pred))

def process_text(text):
    # Remove punctuation

    noPunc = [char for char in text if char not in string.punctuation]
    noPunc = ''.join(noPunc)

    # Make Clean Words
    cleanWords = [word for word in noPunc.split() if word.lower() not in stopwords.words('english')]

    return cleanWords

def load_model(X_test, y_test):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    print(X_test)
    #pred = loaded_model.predict(X_test)

    #print("Accuracy: ", accuracy_score(y_test, pred))


if (path.exists("finalized_model.sav")) == False:
    print("Generating new ML Model")
    messagesBow, df = preprocess_data()
    make_model(messagesBow, df)
    load_model()
else:
    print("Using saved ML Model")
    messagesBow, df = preprocess_data()
    split_data(messagesBow, df)


