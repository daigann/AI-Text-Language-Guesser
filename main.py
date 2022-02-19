# David Mehovic

# import statements used for random number generation, plotting and machine learning libs
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import random

# initialize classifiers
knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()

# here I open all three text files and load them into the appropriate variables
english_file = open("english.txt")
german_file = open("german.txt")
french_file = open("french.txt")

# read the lines for each file
english_lines = english_file.readlines()
german_lines = german_file.readlines()
french_lines = french_file.readlines()

# training dataset used to train the ai
# testing dataset stores the expected values
# target dataset used for testing how well ai performed
training_dataset = []
testing_dataset = []
target_dataset = []
comparative_dataset = []

# go through all english words inside english_lines, find the 5 letter words then split them
# randomly into training dataset and testing dataset. 80% of the words will end up in training
# whereas 20% will end up in testing
# I use a random number between 0 and 9 inclusive. Where if the value returns 8 or 9 its a testing set
# otherwise its a training set
for line in english_lines:
    line = line.replace('\n', '')

    if len(line) == 5:

        # random number determines if this word goes into training dataset or testing dataset
        if random.randint(0, 9) < 8:
            training_dataset.append([ord(char) for char in line])
            target_dataset.append(0)

        else:
            testing_dataset.append([ord(char) for char in line])
            comparative_dataset.append(0)

# go through all german words inside german_lines, find the 5 letter words then split them
# randomly into training dataset and testing dataset. 80% of the words will end up in training
# whereas 20% will end up in testing
# I use a random number between 0 and 9 inclusive. Where if the value returns 8 or 9 its a testing set
# otherwise its a training set
for line in german_lines:
    line = line.replace('\n', ' ')

    if len(line) == 5:
        # random number determines if this word goes into training dataset or testing dataset
        if random.randint(0, 9) < 8:
            training_dataset.append([ord(char) for char in line])
            target_dataset.append(1)

        else:
            testing_dataset.append([ord(char) for char in line])
            comparative_dataset.append(1)

# same as the above two, splitting my french words where 80% go in training and 20% go in testing
# I use a random number between 0 and 9 inclusive. Where if the value returns 8 or 9 its a testing set
# otherwise its a training set
for line in french_lines:
    line = line.replace('\n', ' ')

    if len(line) == 5:
        # random number determines if this word goes into training dataset or testing dataset
        if random.randint(0, 9) < 8:
            training_dataset.append([ord(char) for char in line])
            target_dataset.append(2)

        else:
            testing_dataset.append([ord(char) for char in line])
            comparative_dataset.append(2)

# training models happens here using training dataset and target dataset
knn_model.fit(training_dataset, target_dataset)
svm_model.fit(training_dataset, target_dataset)
mlp_nn.fit(training_dataset, target_dataset)

# the predictions are made here and stored into a np array
# along with the comparative test dataset, I use this to compare the predictions to the actual
# values of what the output should be
knnArray = np.array(knn_model.predict(testing_dataset))
svmArray = np.array(svm_model.predict(testing_dataset))
mlpArray = np.array(mlp_nn.predict(testing_dataset))

compareArray = np.array(comparative_dataset)

# using np.count_nonzero I count how many english, german, and french words there are
# the encoding works as 0 - english, 1 - german, 2 - french. So I am counting how many of each
compareNum_zeros = np.count_nonzero(compareArray == 0)
compareNum_ones = np.count_nonzero(compareArray == 1)
compareNum_twos = np.count_nonzero(compareArray == 2)

# this counts knns predictions
knnNum_zeros = np.count_nonzero(knnArray == 0)
knnNum_ones = np.count_nonzero(knnArray == 1)
knnNum_twos = np.count_nonzero(knnArray == 2)

# here svm's predictions are counted
svmNum_zeros = np.count_nonzero(svmArray == 0)
svmNum_ones = np.count_nonzero(svmArray == 1)
svmNum_twos = np.count_nonzero(svmArray == 2)

# and finally see how mlp performed
mlpNum_zeros = np.count_nonzero(mlpArray == 0)
mlpNum_ones = np.count_nonzero(mlpArray == 1)
mlpNum_twos = np.count_nonzero(mlpArray == 2)

# this next section does some housekeeping to get the percentage error of language prediction
# it finds the errors in each language, subtracts it from the total count to see how many words
# were incorrectly guessed
compareTotals = compareNum_zeros + compareNum_ones + compareNum_twos

# So what I do here is find how many errors there are and subtract it from the total number
# of words to find the percent error. Each incorrect guess subtracts a point
knnEnglishErrors = abs(compareNum_zeros - knnNum_zeros)
knnGermanErrors = abs(compareNum_ones - knnNum_ones)
knnFrenchErrors = abs(compareNum_twos - knnNum_twos)
knnTotalErrors = knnEnglishErrors + knnGermanErrors + knnFrenchErrors
knnCount = compareTotals - knnTotalErrors

# doing the same as above just for svm
svmEnglishErrors = abs(compareNum_zeros - svmNum_zeros)
svmGermanErrors = abs(compareNum_ones - svmNum_ones)
svmFrenchErrors = abs(compareNum_twos - svmNum_twos)
svmTotalErrors = svmEnglishErrors + svmGermanErrors + svmFrenchErrors
svmCount = compareTotals - svmTotalErrors

# again same as above but for mlp
mlpEnglishErrors = abs(compareNum_zeros - mlpNum_zeros)
mlpGermanErrors = abs(compareNum_ones - mlpNum_ones)
mlpFrenchErrors = abs(compareNum_twos - mlpNum_twos)
mlpTotalErrors = mlpEnglishErrors + mlpGermanErrors + mlpFrenchErrors
mlpCount = compareTotals - mlpTotalErrors

# Label text for each graph
labels = ("KNN", "SVM", "MLP")

# Numbers that you want the bars to represent
value = [knnCount / compareTotals * 100,
         svmCount / compareTotals * 100,
         mlpCount / compareTotals * 100]

# Title of the plot
plt.title("Model Accuracy")

# Label for the x values of the bar graph
plt.xlabel("Accuracy")

# Drawing the bar graph
y_pos = np.arange(len(labels))
plt.barh(y_pos, value, align="center", alpha=0.5)
plt.yticks(y_pos, labels)

# Display the graph
plt.show()
