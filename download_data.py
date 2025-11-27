import pandas as pd
import os

if not os.path.exists("data"):
    os.makedirs("data")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(url, names=columns)
df.to_csv("data/diabetes.csv", index=False)

print("Downloaded diabetes.csv to data folder")
