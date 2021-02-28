# -------------------------------------------------------------------------
# AUTHOR: Zewen Lin
# FILENAME: naive_bayes.py
# SPECIFICATION: use naive bayes strategy to predict the result
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5hr
# -----------------------------------------------------------*/

# importing some Python libraries
import csv

from sklearn.naive_bayes import GaussianNB

db = []
# reading the training data
# --> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
X = []
dataset = {'Sunny': 1, 'Overcast': 2, 'Rain': 3, 'Hot': 1, 'Mild': 2,
           'Cool': 3,
           'High': 1, 'Normal': 2, 'Weak': 1, 'Strong': 2, 'No': 1, 'Yes': 2
           }
for i in range(len(db)):
    temp = []
    for j in range(1, len(db[i]) - 1):
        temp.append(dataset[db[i][j]])
    X.append(temp)

# transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
Y = []
for i in range(len(db)):
    Y.append(dataset[db[i][5]])

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the data in a csv file
# --> add your Python code here
test_db = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            test_db.append(row)
test_X = []
for i in range(len(test_db)):
    test_temp = []
    for j in range(1, len(test_db[i]) - 1):
        test_temp.append(dataset[test_db[i][j]])
    test_X.append(test_temp)

# printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(
    15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(
    15) + "Confidence".ljust(15))
# use your test samples to make probabilistic predictions.
for i in range(len(test_db)):
    predicted = clf.predict_proba([test_X[i]])[0]
    for j in range(len(test_db[i]) - 1):
        print(test_db[i][j].ljust(15), end="")
    if predicted[0] > predicted[1]:
        print("No".ljust(15), end="")
        print(predicted[0])
    elif predicted[0] < predicted[1]:
        print("Yes".ljust(15), end="")
        print(predicted[1])