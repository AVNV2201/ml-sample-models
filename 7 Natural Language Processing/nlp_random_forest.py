#import libraries
import pandas as pd
import numpy as np

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting=3 )

#clean text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',  dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer( max_features=1000 )
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# prepare test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0 )

# #prepaer classification model 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators=500, criterion='entropy', random_state = 0 )
classifier.fit(X_train,y_train)

# prediction for test set
#import libraries
import pandas as pd
import numpy as np

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting=3 )

#clean text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',  dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer( max_features=1000 )
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# prepare test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0 )

# #prepaer classification model 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier( criterion = 'entropy' )
classifier.fit(X_train,y_train)

# prediction for test set
y_pred = classifier.predict(X_test)

# get the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)





























