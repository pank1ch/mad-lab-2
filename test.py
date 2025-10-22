import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("Titanic-Dataset.csv")

# пропуски
print(df.isna().sum())

# Удаление Cabin оч много пропусков
df = df.drop(columns=['Cabin'])

# Удаление строк с пропусками age и embarked
df = df.dropna()

print(df.isna().sum())
print(df.shape)

# === Дескриптивный анализ ===

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

print(df[numeric_cols].describe())

for col in numeric_cols:
    plt.figure(figsize=(5, 3))
    plt.hist(df[col], bins='sturges', edgecolor='black')
    plt.title(f'Распределение признака: {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.show()

    #шапиро
    stat_sw, p_sw = shapiro(df[col])

    #дагостино
    stat_nt, p_nt = normaltest(df[col])

    print(f"{col}: Shapiro p = {p_sw:.5f} | D’Agostino p = {p_nt:.5f}")

# === Дерево решений ===

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = df[['Survived'] + features]

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df.drop('Survived', axis=1)
y = df['Survived']

# обучающя и test выборка (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Оценка точности
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(acc, 2))

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# Визуализация дерева
plt.figure(figsize=(32, 16))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Не выжил', 'Выжил'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Дерево решений для классификации выживания пассажиров")
plt.show()
