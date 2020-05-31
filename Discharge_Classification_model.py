
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("BatteryDischargeClass.csv")
df = pd.DataFrame(data)
print(df.columns)
X = df.loc[:,['voltage','temperature']]
print(X.columns)
Y = df.loc[:,['State']]

# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


print(model)


# make predictions for test data
y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))









