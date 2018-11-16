# Logistic Regression Classifier [Un-Normalized]

# Importing reqd. Libraries
import numpy as np
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from pandas import read_csv 
nltk.download("stopwords")

def write_lines(outfile,lists): 
    f = open(outfile, "a", encoding='utf-8')
    for lines in lists:
        lines=str(lines)+"\n"
        f.writelines(str(lines))
        
       
    f.close()

outfile = "results.txt"
lists = []


# Importing and Cleaning the dataset
dataset = read_csv("tweets-train.csv")
X = dataset.iloc[:, [0]].values
X = np.ndarray.tolist(X)  
y = dataset.iloc[:, [1]].values
y = np.ndarray.tolist(y) 
y_corpus =[]

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
sum = np.sum(y)

# Pickling the Dataset [Persisting the dataset] 
with open("X.pickle", "wb") as f:
    pickle.dump(X, f)                                                  
with open("y.pickle", "wb") as f:
    pickle.dump(y, f) 
         
# Pre-processing the dataset, Normalizing the Tweets and Creating the corpus 
corpus = []
for i in range(len(X)):
    origTweet = str(X[i])
    corpus.append(origTweet)
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
text_Train, text_Test, sd_Train, sd_Test = train_test_split(corpus, y, test_size=0.25, random_state=0)

# Creating TF-IDF Model using TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# tVectorizer = TfidfVectorizer(max_features=1000, min_df=3, max_df=0.6, stop_words=stopwords.words("english"))
# tVectorizer = TfidfVectorizer(max_features=1000, min_df=3, max_df=0.6)
tVectorizer = TfidfVectorizer(ngram_range = (1, 3))
# tVectorizer = CountVectorizer(max_features=1000, min_df=3, max_df=0.6, stop_words=stopwords.words("english"))
# tVectorizer = CountVectorizer(max_features=1000, min_df=3, max_df=0.6)
# tVectorizer = CountVectorizer()
text_Train = tVectorizer.fit_transform(text_Train)

# Random Over-sampling of the majority class
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
text_Train, sd_Train = ros.fit_sample(text_Train, sd_Train)

text_Test = tVectorizer.transform(text_Test)

# Fitting classifier to the Training set &  Predicting the Test set results
from sklearn.linear_model import LogisticRegression

# Logistic Regression
clf1 = LogisticRegression(random_state=0)
clf1.fit(text_Train, sd_Train)
sd_Pred1 = clf1.predict(text_Test)


# Visualizing the RESULTS

# Making the Confusion Matrix(Classification Evaluation Metric)
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(sd_Test, sd_Pred1)


# Generating the Classification Report(Classification Evaluation Metric)
from sklearn.metrics import classification_report
report_lr = classification_report(sd_Test, sd_Pred1)

# Area Under ROC Curve
import matplotlib.pyplot as plt
from sklearn import metrics
y_pred_proba = clf1.predict_proba(text_Test)[::,1]
fpr, tpr, _ = metrics.roc_curve(sd_Test,  y_pred_proba)
auc = metrics.roc_auc_score(sd_Test, y_pred_proba)
plt.plot(fpr,tpr,label="svm, auc="+str(auc))
plt.legend(loc=4)
plt.show()

str1="Accuracy for lr with Un-Normalized"
lists.append(str1)
lists.append(report_lr)
str2 = "Confusion Matrix for lr"
lists.append(str2)
lists.append(cm_lr)

write_lines(outfile, lists)
