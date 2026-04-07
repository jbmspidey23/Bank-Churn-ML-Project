import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



class BankChurnPipeline:

    def __init__(self, url):

        self.url = url
        self.model = LogisticRegression(max_iter=1000)
        self.df = None

    def load_and_clean(self):

        print("Scrubbing the dataset...")
        self.df = pd.read_csv(self.url)
        self.df = self.df.drop(columns=["RowNumber", "CustomerId", "Surname"])
        self.df['Gender'] = self.df['Gender'].map({"Male": 0, "Female": 1})
        self.df[['Age', 'Balance']] = self.df[['Age', 'Balance']].fillna(self.df[['Age', 'Balance']].median())
        print("Data is squeaky clean!")

    def train(self):

        print("Training the model...")
        x = self.df[['Gender', 'Age', 'Balance', 'NumOfProducts', 'CreditScore']]
        y = self.df['Exited']


        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.model.fit(x_train, y_train)
        print("Model trained!")

    def evaluate(self):

        predictions = self.model.predict(self.x_test)
        score = accuracy_score(self.y_test, predictions)
        print(f"\n--- Final Exam Score ---")
        print(f"Accuracy: {score * 100:.2f}%")

    def save_model(self):
        joblib.dump(self.model, "bank_brain.pkl")
        print("\n--- Model Saved! ---")
        print("The brain has been successfully downloaded to your hard drive.")


if __name__ == "__main__":
    data_link = "https://raw.githubusercontent.com/shivang98/Churn-Modelling/master/Churn_Modelling.csv"


    my_pipeline = BankChurnPipeline(data_link)


    my_pipeline.load_and_clean()
    my_pipeline.train()
    my_pipeline.evaluate()
    my_pipeline.save_model()