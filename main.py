import pandas as pd
#load data
df = pd.read_csv("titanic.csv")
print(df.head())
#clean data
print("--Missing data before cleaning--")
print(df.isnull().sum())
df = df.drop(columns=['Cabin','Ticket','Name'])
df['Age'] = df['Age'].fillna(df['Age'].median())
print("--Missing data after cleaning")
print(df.isnull().sum())
df["Sex"] = df['Sex'].map({'male':0,'female':1})
print("\n--- Sex column after translation ---")
print(df['Sex'].head())
x = df[["Pclass","Sex","Age","Fare"]]
y=df["Survived"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =42)
print("\n--- Data Split Complete ---")
print(f"Training data: {x_train.shape} rows")
print(f"Testing data: {x_test.shape} rows")
#training data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
print("\n--- Model Training Complete ---")
print("The model has officially learned the patterns!")
#testing data
from sklearn.metrics import accuracy_score
predictions = model.predict(x_test)
score = accuracy_score(y_test,predictions)
print("\n--- Final Exam Score ---")
print(f"Accuracy: {score * 100:.2f}%")
