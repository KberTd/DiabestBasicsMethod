import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier


diabetes = pd.read_csv("C:\\Users\\ARDA\\PycharmProjects\\pythonProjectybs.py\\diabetes.csv")


X = diabetes.drop("Outcome", axis = 1) # axis = 0 : satır //// axis = 1 : sütun
y = diabetes["Outcome"]

#
# print (X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

sc= StandardScaler() # nested array dönüyor.
                     # z = (x - u) / s
                     # z = new value after calculation
                     # x = old value
                     # u = mean
                     # s = standard deviation
#
# print(X_train)
# print (X_train)
X_train = sc.fit_transform(X_train) # nested array dönüyor. fit_transform does the math to fit all the values
                                    # so that all the variables affect the model at the same level.
# print (X_train)

X_test = sc.transform(X_test) # we just transformed because we didn't want to calculate
                              # mean and standard deviation again.

#
# print (X_train)


clf = svm.SVC()

clf.fit(X_train, y_train) # X_train ve y_train datalarını al ve SVC modeline uyumlandır.
pred_clf = clf.predict(X_test)
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))



mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=100)
                    #3 hidden layers and size of each are 11
                    #maximum iterations is 500. It means, it will go through the data 500 times.

mlpc.fit(X_train,y_train)
pred_mlpc = mlpc.predict(X_test)

# print (classification_report(y_test, pred_mlpc))
# print(confusion_matrix(y_test, pred_mlpc))

a = [[6,148,72,35,0,33.6,0.627,50]]

Xnew = sc.transform(a)
ynew1 = clf.predict(Xnew)
ynew2 = mlpc.predict(Xnew)
print(ynew1, ynew2)
