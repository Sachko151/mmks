import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

data = pd.read_csv("gesture_dataset.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2
)

model = SVC(kernel="rbf")

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

print("Accuracy:",accuracy)

pickle.dump(model,open("gesture_model.pkl","wb"))