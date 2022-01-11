import pandas as pd
import numpy as np

#load and sanitise the data
w = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
col_names = w.columns[0].replace('"', '').split(';')
w = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', names = col_names, header = None)
w.drop(0, inplace = True)
w.reset_index(drop = True, inplace = True)
for entry in w.index:
    w.iloc[entry] = [float(i) for i in w.iloc[entry,0].split(';')]
w['fixed acidity'] = [float(i) for i in w.iloc[:,0]]

#define wines above 7 as good. create 2 bins for good and bad wines distinguished by cut-off at quality = 7
bins = [2,7,10]
labels = ['bad', 'good']
w['quality'] = pd.cut(w['quality'], bins = bins, labels = labels)

#encode the response variable 
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
w['quality'] = lbl.fit_transform(w['quality'])

#separate data from the dataframe into input and output
X = w.drop('quality', axis = 1)
y = w['quality']

#split data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 40)

#normalise the input data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#creating models
#model 1: RandomForrestClassifier
from sklearn.ensemble import RandomForestClassifier
rcf = RandomForestClassifier(n_estimators = 200)
rcf.fit(X_train_scaled, y_train)
y_pred_rcf = rcf.predict(X_test_scaled)
from sklearn.metrics import accuracy_score
score_rcf = accuracy_score(y_test,y_pred_rcf)

#model 2: support vector classification
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test)
score_svc = accuracy_score(y_test, y_pred_svc)


#model 3: MLPClassifier
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(random_state = 10)
mlpc.fit(X_train_scaled, y_train)
y_pred_mlpc = mlpc.predict(X_test_scaled)
score_mlpc = accuracy_score(y_test,y_pred_mlpc)

#print accuracy scores
print("RandonForestClassifier: {}".format(score_rcf))
print("SupportVectorClassifier: {}".format(score_svc))
print("MLPClassifier: {}".format(score_mlpc))