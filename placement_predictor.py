import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
dataset = pd.read_csv(r"C:\Users\ANUBHAB\OneDrive\Documents\IIT_Placement_Dataset_Updated.csv")
x = dataset.iloc[:,:-1]
y = dataset["Placement_Package_LPA"]
lr = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=47)
lr.fit(x_train,y_train)
#cgpa = float(input("Enter your CGPA = "))
#iq = float(input("Enter your IQ level = "))
#user_input = [[iq,cgpa]]
#print(lr.predict(user_input))
#y_predict = lr.predict(x)
#print(lr.score(x_test,y_test))
#print(dataset)
#sns.heatmap(data= dataset.corr(),annot= True)
#plt.show()

import os

# 1. Get the path to the folder where this python file (placement_predictor.py) is located
current_directory = os.path.dirname(__file__)

# 2. Combine that path with your folder and file name
file_path = os.path.join(current_directory, 'savedmodel', 'model.joblib')

# 3. Save the model safely
joblib.dump(lr, file_path)
print(f"Model saved successfully at: {file_path}")






