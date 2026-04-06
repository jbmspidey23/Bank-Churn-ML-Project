import pandas as pd
# Pull the data directly from the internet
url = "https://raw.githubusercontent.com/shivang98/Churn-Modelling/master/Churn_Modelling.csv"
df = pd.read_csv(url)
print(df.head())
#cleaning the data
print(df.isnull().sum())
df = df.drop(columns =["RowNumber","CustomerId","Surname"])
df['Gender'] = df['Gender'].map({"Male": 0, "Female": 1})
df[['Age', 'Balance']] = df[['Age', 'Balance']].fillna(df[['Age', 'Balance']].median())
#split data
x=df[['Age','Balance','NumOfProducts','CreditScore','Gender']]
y=df['Exited']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("\n--- Data Split Complete ---")
print(f"Training Data:{x_train.shape}")
print(f"testing data:{x_test.shape}")
#training data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
print("\n--Data Training completed")
print("The model has officially learned the patterns!")
#testing Data
from sklearn.metrics import accuracy_score
predictions = model.predict(x_test)
score = accuracy_score(y_test,predictions)
print("\n--Data Testing Complete--")
print(f"Accuracy:{score*100:.2f}%")