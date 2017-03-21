import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.metrics import accuracy_score
import random
from sklearn import preprocessing

filename = "letter-recognition.data"
raw_data = open(filename, 'rb')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x)

#split data and labels
X_data = data[:,1:]
Y_data = data[:,0]

#preprocessing: normalization, 0 mean and scaling variance(from -1 to 1)
#X_data = preprocessing.normalize(X_data, norm='l2')
X_data = preprocessing.scale(X_data)

#percentage of train and test data 
perc = 0.75;
no_of_points = X_data.shape[0]
total_perc = perc*no_of_points
print(perc*no_of_points)

#train data
Y_train = Y_data[:total_perc]
X_train = X_data[:total_perc]
X_train = X_train.astype('float')


#test data 
Y_test = Y_data[(total_perc+1):]
X_test = X_data[(total_perc+1):]
X_test = X_test.astype('float')

#train classifier 
clf = GaussianNB()
clf.fit(X_train, Y_train)

#prediction
y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)


print(accuracy)
