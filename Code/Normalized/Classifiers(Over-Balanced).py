# Classifiers [Over-Balanced]

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
    tweet = re.sub(r"\W", " ", str(X[i])) # Removing non-word characters
    tweet = tweet.lower() # Converting into lower case
    tweet = re.sub(r"\s+[a-z]\s+", " ", tweet) # Removing single characters in the corpus 
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+", " ", tweet) # Removing multi-spaces by a single space
    # corpus.append(tweet)
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Logistic Regression
clf1 = LogisticRegression(random_state=0)
clf1.fit(text_Train, sd_Train)
sd_Pred1 = clf1.predict(text_Test)

clf2 = SVC(kernel = "linear", random_state = 0)
clf2.fit(text_Train, sd_Train)
sd_Pred2 = clf2.predict(text_Test) 

# Decison Tree Classifier
clf3 = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
clf3.fit(text_Train, sd_Train)
sd_Pred3 = clf3.predict(text_Test) 

# Random Forest Classifier
clf4 = RandomForestClassifier(n_estimators=300, criterion = "entropy", random_state=0)
clf4.fit(text_Train, sd_Train)
sd_Pred4 = clf4.predict(text_Test) 

# Voting Classifier
eclf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2), ('dtc', clf3), ('rf', clf4)], voting='hard')
# eclf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2), ('rf', clf4)], voting='hard')
eclf.fit(text_Train, sd_Train)
sd_Pred = eclf.predict(text_Test)

# Visualizing the RESULTS

# Making the Confusion Matrix(Classification Evaluation Metric)
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(sd_Test, sd_Pred1)
cm_svm = confusion_matrix(sd_Test, sd_Pred2)
cm_dtc = confusion_matrix(sd_Test, sd_Pred3)
cm_rf = confusion_matrix(sd_Test, sd_Pred4)
cm_eclf = confusion_matrix(sd_Test, sd_Pred)

# Generating the Classification Report(Classification Evaluation Metric)
from sklearn.metrics import classification_report
report_lr = classification_report(sd_Test, sd_Pred1)
report_svm = classification_report(sd_Test, sd_Pred2)
report_dtc = classification_report(sd_Test, sd_Pred3)
report_rf = classification_report(sd_Test, sd_Pred4)
report_eclf = classification_report(sd_Test, sd_Pred)

# F_Score plot
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
f1_lr = f1_score(sd_Test, sd_Pred1, average='macro')
f1_svm = f1_score(sd_Test, sd_Pred2, average='macro')
f1_dtc = f1_score(sd_Test, sd_Pred3, average='macro')
f1_rf = f1_score(sd_Test, sd_Pred4, average='macro')
f1_eclf = f1_score(sd_Test, sd_Pred, average='macro')

y_plot = np.array([ f1_dtc, f1_rf, f1_eclf, f1_lr, f1_svm])
x_plot = np.array([20, 40, 60, 80, 100])
plt.scatter(x_plot, y_plot)
plt.scatter(x_plot, y_plot, color = "blue")
plt.plot(x_plot, y_plot, color = "fuchsia")
plt.xticks([20, 40, 60, 80, 100], ["dtc", "rf", "eclf", "lr", "svm"])
plt.xlabel("Classifiers")
plt.ylabel("F1_Score")
plt.show()

str1="Accuracy for lr with tfidf n-gram in 1,3"
lists.append(str1)
lists.append(report_lr)
str2 = "Confusion Matrix for lr"
lists.append(str2)
lists.append(cm_lr)

str1="Accuracy for svm with tfidf n-gram in 1,3"
lists.append(str1)
lists.append(report_svm)
str2 = "Confusion Matrix for svm"
lists.append(str2)
lists.append(cm_svm)

str1="Accuracy for dtc with tfidf n-gram in 1,3"
lists.append(str1)
lists.append(report_dtc)
str2 = "Confusion Matrix for dtf"
lists.append(str2)
lists.append(cm_dtc)

str1="Accuracy for rf with tfidf n-gram in 1,3"
lists.append(str1)
lists.append(report_rf)
str2 = "Confusion Matrix for rf"
lists.append(str2)
lists.append(cm_rf)

str1="Accuracy for eclf with tfidf n-gram in 1,3"
lists.append(str1)
lists.append(report_eclf)
str2 = "Confusion Matrix for rf"
lists.append(str2)
lists.append(cm_eclf)

# Pickling or saving our Classifier
with open("classifier.pickle", "wb") as f:
    pickle.dump(eclf, f)

# Pickling or saving our TfidfVectorizer
with open("tfidf_vectorizer.pickle", "wb") as f:
    pickle.dump(tVectorizer, f)
    
write_lines(outfile, lists)

