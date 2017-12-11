#
# Code by Rajiv Mantena for CS6350 Machine Learning Fall 2016
# Homework6 : Stochastic Gradient
# University of Utah
# 6th of December
# mantenarajiv@gmail.com
#

import numpy
import csv
import sys
# import matplotlib.pyplot


# Function to read the CSV into Numpy.Array
def read_data_csv(matrix):
    new_array = numpy.zeros((len(matrix), 149))
    for count in range(len(matrix)):
        for a in range(len(matrix[count])):
            new_array[count][a] = matrix[count][a]
            # new_array[count][a] = int(''.join(matrix[count][a]))
    return new_array


def read_data(matrix):                          # Function to convert data from list format to a Numpy Array
    flag = 0                                    # Flag for setting if the bit has to be read or not
    feature_var = 0                             # Variable to save the bit value
    new_array = numpy.zeros((len(matrix), 125))  # Initialize a zeros Numpy Array
    for count in range(len(matrix)):            # For each record in the data set
        new_array[count][1] = 1                 # Set a bit to compensate for bias term
        new = zip(*matrix[count])               # Transpose the record for easier fetching of each element.
        value = ''.join(new[0])                 # Trick to get the value out of each row of the transposed record array
        if value == '+':         # If the label of the record is positive change it to 1 (To make my life easier later)
            new_array[count][0] = 1                             # Update the new Numpy array with this data
        else:
            new_array[count][0] = -1                             # Update the new Numpy array with this data
        # I am dividing the array into bits, like shown below.
        # new = ['+','1',' ','3',':','1',' ','4',':','1',' ', ... etc.]
        # Hence, in this data, each bit has to be read and understood
        for a in range(2, len(new)):             # For each bit in the new array
            if ''.join(new[a]) == ' ':          # If the bit is a space, the next bit would be a feature index
                flag = 1
                feature_var = 0
            elif ''.join(new[a]) == ':':        # If the bit is a :, then the feature index has ended.
                flag = 0
                new_array[count][feature_var+1] = 1     # Saving the feature_index into the new matrix
            elif flag == 1:                     # Else, it is the feature number and has to be read
                feature_var = feature_var*10 + int(float(''.join(new[a])))
    return new_array                            # Return the Numpy Array


# SVM function
def svm(data, gamma0, epochs, C):
    w = numpy.zeros(146)
    for each_epochs in range(epochs):
        numpy.random.shuffle(data)                  # Shuffle the data before start of each epoch
        for count in range(len(data)):
            data_here = data[count]  # Copy 2D array into a 1D array
            y_here = data_here[148]  # Separate label,
            x_here = data_here[1:147]  # Separate features,
            weekday = data_here[0]  # Separate the Weekday boolean. 	(Not using this as a feature yet)
            if weekday:
                gamma_t = gamma0/float(1+gamma0*count/C)
                product = y_here*numpy.dot(w, x_here)
                if product <= 1:
                    w = (1-gamma_t)*w + gamma_t*C*y_here*x_here
                else:
                    w *= (1-gamma_t)
    return w


# SGD function
def sgd(data, gamma0, epochs, sigma):
    w = numpy.zeros(146)
    for each_epochs in range(epochs):
        # myCount = []
        # myGradient = []
        # logLikely = []
        numpy.random.shuffle(data)                  # Shuffle the data before start of each epoch
        for count in range(len(data)):
            data_here = data[count]  # Copy 2D array into a 1D array
            y_here = data_here[148]  # Separate label,
            x_here = data_here[1:147]  # Separate features,
            weekday = data_here[0]  # Separate the Weekday boolean. (Not using this as a feature yet)
            if weekday:
                gamma_t = gamma0/float(1+gamma0*count/sigma)
                product = y_here*numpy.dot(w, x_here)
                gradient = (y_here*x_here)/float(numpy.exp(product)+1)-2*w/float(sigma*sigma)
                w += gamma_t*gradient
            # myGradient.append(gradient)
            # logLikely.append(numpy.log(1+numpy.exp(-1*product)))
            # myCount.append(count)
    # if each_epochs == 5:
        # plt.plot(myCount, logLikely)
        # plt.ylabel('LogLikely')
        # plt.xlabel('Iterations')
        # plt.show()
        # print w
    return w


# Function to find the accuracy of the final_vector
def validate(final_vector, test_array):
    correct_predictions = 0                             # Variable to save the number of corrections
    tp = 0                                              # True Positive variable
    fp = 0                                              # False Positive variable
    fn = 0                                              # False Negative variable
    total_predictions = len(test_array)                 # Number of records ?
    for current_prediction in range(total_predictions): # loop over all the records
        y_here = test_array[current_prediction][0]
        x_here = test_array[current_prediction][1:125]  # Ignore the 0th element of
        predicted = y_here*numpy.dot(final_vector, x_here)      # What is the prediction of W (final vector) ?
        if predicted >= 0:
            correct_predictions += 1
        if y_here >=0 and numpy.dot(final_vector, x_here) >= 0:
            tp += 1
        elif y_here <0 and numpy.dot(final_vector, x_here) >= 0:
            fp += 1
        elif y_here >=0 and numpy.dot(final_vector, x_here) < 0:
            fn += 1
    correct_predictions = (correct_predictions/float(total_predictions))*100
    return correct_predictions, tp, fp, fn


# Function to find the accuracy of the final_vector
def validate_winnow(final_vector, test_array):
    theta = 147
    correct_predictions = 0                             # Variable to save the number of corrections
    total_predictions = len(test_array)                 # Number of records ?
    total_predictions_weekdays = 0
    for current_prediction in range(total_predictions): # loop over all the records
        y_here = test_array[current_prediction][148]
        x_here = test_array[current_prediction][1:147]  # Ignore the 0th element of the array. (Label)
        weekday = test_array[current_prediction][0]
        if weekday:
            predicted = y_here * numpy.dot(final_vector, x_here)  # What is the prediction of W (final vector) ?
            # print "Predicted : ", predicted
            if predicted >= 0:
                correct_predictions += 1                # Increment correct_predictions
            total_predictions_weekdays += 1
    # print "correct_predictions : ", correct_predictions, "total_predictions_weekdays : ", total_predictions_weekdays
    correct_predictions = (correct_predictions/float(total_predictions_weekdays))*100    # Convert to average
    return correct_predictions                          # Return predictions.


# Main function
def main():
    print "Running the stochastic gradient algorithm on train and test data (after cross-validation): "
    gamma0 = 0.005
    epochs = 30
    number_of_tests = 35
    sigma = 80
    accuracies = []

    # Reading training file
    my_data_file = open('data/Data_Training_e.csv')            # Fetching and saving it as a list
    with my_data_file as f:
        data = csv.reader(f)
        my_data_matrix = list(data)
    array = read_data_csv(my_data_matrix)               # Converting the list data into a Numpy Array

    # Reading test file
    my_data_file = open('data/Data_Test_e.csv')            # Fetching and saving it as a list
    with my_data_file as f:
        data = csv.reader(f)
        my_data_matrix = list(data)
    test_array = read_data_csv(my_data_matrix)               # Converting the list data into a Numpy Array

    for count in range(number_of_tests):
        final_vec = sgd(array, gamma0, epochs, sigma)
        accuracy = validate_winnow(final_vec, array)
        accuracies.append(accuracy)                                     # Saving the obtained accuracy
        sys.stdout.write("\r       Working on record : %d " % count)          # Code to just display the progress
        sys.stdout.write("out of %d tests" % number_of_tests)
        sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.flush()
    print "       Number of tests performed are :", number_of_tests, "                          "
    print "       gamma(0) is set at :", gamma0
    print "       Sigma used for these tests :", sigma, "(This was found the best sigma based on cross validation)"
    print "       The number of epochs for each test is set at :", epochs
    print "       ------ Validating the TRAIN Dataset, we get the following accuracy ------"
    print "       Average of accuracies obtained is :", "%.3f" % numpy.average(accuracies), "% with a Standard Deviation of", \
        "%.3f" %numpy.std(accuracies)

    accuracies = []

    for count in range(number_of_tests):
        final_vec = sgd(array, gamma0, epochs, sigma)
        accuracy = validate_winnow(final_vec, test_array)
        accuracies.append(accuracy)                                     # Saving the obtained accuracy
        sys.stdout.write("\r       Working on record : %d " % count)          # Code to just display the progress
        sys.stdout.write("out of %d tests" % number_of_tests)
        sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.flush()
    print "       ------ Validating the TEST Dataset, we get the following accuracy ------"
    print "       Average of accuracies obtained is :", "%.3f" % numpy.average(accuracies), "% with a Standard Deviation of", \
        "%.3f" %numpy.std(accuracies)
    # return 0

if __name__ == '__main__':
    main()