# -------------------------------------------------------------------------
# AUTHOR: Zewen Lin
# FILENAME: knn.py
# SPECIFICATION: calculate the leave-one-out cross-validation error rate (LOO-CV) for 1NN
# FOR: CS 4210-Assignment #2
# TIME SPENT: 1hr
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
error = 0
total_predict = 0

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    X = []
    for j in range(len(db)):
        temp = []
        temp_check = []
        for k in range(len(db[j])):
            temp_check.append(db[j][k])
        if temp_check != instance:
            for k in range(len(temp_check) - 1):
                temp.append(temp_check[k])
            X.append(temp)
    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    Y = []
    for j in range(len(db)):
        if db[j] != instance:
            if db[j][2] == '-':
                Y.append(1)
            elif db[j][2] == '+':
                Y.append(2)
    #store the test sample of this iteration in the vector testSample
    testSample = instance

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([[testSample[0], testSample[1]]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    total_predict += 1
    if class_predicted == 1:
        error += 1

#print the error rate
error_rate = error / total_predict
print("Error rate: " + str(error_rate))






