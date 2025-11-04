import sys
sys.stdout.reconfigure(encoding='utf-8')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("Titanic-Dataset.csv")

# слишком много пропущенных значений
df = df.drop(columns=['Cabin'])

df = df.dropna()


df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked'])

# one-hot encoding

df = pd.get_dummies(df, columns=['Sex'], drop_first=True)


X = df.drop('Survived', axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
scaler.fit(X_train)  
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)


y_pred = mlp.predict(X_test_scaled)


print("Accuracy (точность):", accuracy_score(y_test, y_pred))
print("Confusion matrix (матрица ошибок):\n", confusion_matrix(y_test, y_pred))
print("Classification report (точность, полнота и F1 по каждому классу):\n", classification_report(y_test, y_pred))
