# -------------------------------------------------------------------------
# AUTHOR: Zewen Lin
# FILENAME: decision_tree.py
# SPECIFICATION: decision tree
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2hr
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv',
            'contact_lens_training_3.csv']
dict_dataset = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3,
                'Myope': 1, 'Hypermetrope': 2,
                'No': 1, 'Yes': 2,
                'Normal': 1, 'Reduced': 2
                }
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for i in range(len(dbTraining)):
        temp = []
        for j in range(len(dbTraining[i]) - 1):
            temp.append(dict_dataset[dbTraining[i][j]])
        X.append(temp)

    # transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for i in range(len(dbTraining)):
        Y.append(dict_dataset[dbTraining[i][4]])

    # loop your training and test tasks 10 times here
    accuracy = []
    for i in range(10):
        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)
        # read the test data and add this data to dbTest
        dbTest = []
        test_data = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0:
                    dbTest.append(row)
        correct_predict = 0
        total_data = len(dbTest)
        for data in dbTest:
            temp_test = []
            for inner_data in range(len(data) - 1):
                temp_test.append(dict_dataset[data[inner_data]])
            class_predicted = clf.predict([temp_test])[0]
            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_predicted == dict_dataset[data[4]]:
                correct_predict += 1
        accuracy.append(correct_predict / total_data)
    # print the lowest accuracy of this model during the 10 runs (training and test set).
    print("final accuracy when training on " + ds + " : " + str(min(accuracy)))

