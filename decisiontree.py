from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
s=pd.read_csv('loan_data.csv')

dt = pd.get_dummies(s,columns=['purpose'],drop_first=True)

# print(dt.head())
# print(s.info())
# print(dt.info())
x = dt.drop("not.fully.paid",axis=1)
y = dt["not.fully.paid"]
x_trained, x_test , y_trained, y_test = train_test_split(x,y,test_size=.4,random_state=101)

o=DecisionTreeClassifier()#creation of object
o.fit(x_trained,y_trained)
per = o.predict(x_test)

print(per)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test,per))
print(classification_report(y_test,per))


