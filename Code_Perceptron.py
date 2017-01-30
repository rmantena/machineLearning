#
# Code by Rajiv Mantena & Devi Ayyagari for CS6350 Machine Learning Fall 2016
# Final Project
# University of Utah
# 7th of November
# mantenarajiv@gmail.com
# akdevi.92@gmail.com
#

import numpy
import csv
import sys

# Function to read the CSV into Numpy.Array
def read_data_csv(matrix):
    new_array = numpy.zeros((len(matrix),150))
    for count in range(len(matrix)):
        new_array[count][0] = matrix[count][0]
        new_array[count][1] = 1
        for a in range(1,len(matrix[count])):
            new_array[count][a+1] = matrix[count][a]
            # new_array[count][a] = int(''.join(matrix[count][a]))
    return new_array

# Preceptron Algorithm implementation
def simple_preceptron(data, learning_rate):
    w = numpy.random.rand(147)                      # Initializing a random vector
    w = w/float(numpy.linalg.norm(w))
    margin = 1
    corrections = 0                                 # Variable to count the number of mistakes that would be made
    for count in range(len(data)):                  # For each record in the data set
        data_here = data[count]                     # Fetching the data from the NP-ND-Array into ...
        y_here = data_here[149]
        x_here = data_here[1:148]
        weekday = data_here[0]
        dot_prod = numpy.dot(w, x_here)             # Find W^T*x_i
        if weekday:
            if y_here == 0:                         # Updating to get correct (y_i)(W^T)(x_i)
                dot_prod = -1*dot_prod
            if dot_prod <= margin:                       # if (y_i)(W^T)(x_i) <= 0 ???
                corrections += 1                    # Mistake, update the no of mistakes
                if y_here == 1:                     # Update the W based on the y_here label (+ or -)
                    w = w + learning_rate*x_here    #
                else:
                    w = w - learning_rate*x_here
    return w, corrections                           # return the number of corrections and the final weight vector


# Aggressive Preceptron Algorithm implementation
def aggressive_preceptron(data, learning_rate):
    w = numpy.random.rand(147)                      # Initializing a random vector
    w = w/float(numpy.linalg.norm(w))
    margin = 1
    corrections = 0                                 # Variable to count the number of mistakes that would be made
    for count in range(len(data)):                  # For each record in the data set
        data_here = data[count]                     # Fetching the data from the NP-ND-Array into ...
        # print data_here
        y_here = data_here[149]
        x_here = data_here[1:148]
        weekday = data_here[0]
        dot_prod = numpy.dot(w, x_here)             # Find W^T*x_i
        if weekday:
            if y_here == 0:                         # Updating to get correct (y_i)(W^T)(x_i)
                dot_prod = -1*dot_prod
            if dot_prod <= margin:                  # if (y_i)(W^T)(x_i) <= 0 ???
                corrections += 1                    # Mistake, update the no of mistakes
                if y_here == 1:                     # Update the W based on the y_here label (+ or -)
                    eta = (margin - numpy.dot(w, x_here))\
                                    /(numpy.dot(x_here.transpose(), x_here)+1)
                    w = w + eta*x_here              # Aggressive update rules
                else:
                    eta = (margin + numpy.dot(w, x_here))\
                                    /(numpy.dot(x_here.transpose(), x_here)+1)
                    w = w - eta*x_here
    return w, corrections                           # return the number of corrections and the final weight vector


# Function to find the accuracy of the final_vector
def validate_perceptron(final_vector, test_array):
    theta = 148
    correct_predictions = 0                             # Variable to save the number of corrections
    total_predictions = len(test_array)                 # Number of records ?
    total_predictions_weekdays = 0
    for current_prediction in range(total_predictions): # loop over all the records
        y_here = test_array[current_prediction][149]
        x_here = test_array[current_prediction][1:148]  # Ignore the 0th element of the array. (Label)
        weekday = test_array[current_prediction][0]
        if weekday:
            predicted = numpy.dot(final_vector,x_here)  # What is the prediction of W (final vector) ?
            if predicted >= 0 and y_here == 1:      # Does it match the label ?
                correct_predictions += 1                # Increment correct_predictions
            elif predicted < 0 and y_here == 0:     # Same as above
                correct_predictions += 1                # Increment correct_predictions
            total_predictions_weekdays += 1
    correct_predictions = (correct_predictions/float(total_predictions_weekdays))*100    # Convert to average
    return correct_predictions                          # Return predictions.

# Main function
def main():
    learning_rate = 1                                   # Initializing learning rate
    # Variable to define the algorithm to run
    # Select 1 for Perceptron
    # Select 2 for Aggressive Perceptron
    Algorithm_Select = 2
    print "Running the Preceptron/Winnow code : "
    my_data_file = open('Data_Training.csv')            # Fetching and saving it as a list
    with my_data_file as f:
        data = csv.reader(f)
        my_data_matrix = list(data)
    array = read_data_csv(my_data_matrix)               # Converting the list data into a Numpy Array

    my_test_file = open('Data_Test.csv')                # Fetching and saving it as a list
    with my_test_file as f:
        data = csv.reader(f)
        my_test_matrix = list(data)
    test_array = read_data_csv(my_test_matrix)                          # Converting the list data into a Numpy Array
    num_of_tests = 10                                                   # Defining the number of test to be performed
    corrections = []                                                    # Initializing arrays to compute accuracy and mistakes
    accuracies = []
    for count in range(num_of_tests):                                   # Loop over the number of tests
        if (Algorithm_Select == 1):
            (final_vec, correction) = simple_preceptron(array, learning_rate)       # Calling the preceptron function
            curr_acc = validate_perceptron(final_vec, test_array)                   # Validating the obtained vector
        elif (Algorithm_Select == 2):
            (final_vec, correction) = aggressive_preceptron(array, learning_rate)   # Calling the preceptron function
            curr_acc = validate_perceptron(final_vec, test_array)                   # Validating the obtained vector
        else:
            curr_acc = 0
            correction = 0
        accuracies.append(curr_acc)                                     # Saving the obtained accuracy
        corrections.append(correction)                                  # and correction
        sys.stdout.write("\r       Working on record : %d " % count )   # Code to just display the progress
        sys.stdout.write("out of %d tests" % num_of_tests )
        sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.flush()
    # Code to print out the experimental statistics at the end of the run
    print "       Number of tests performed are :", num_of_tests, "                          "
    print "       Learning rate is set at :", learning_rate
    print "       Average number of mistakes done are :", numpy.average(corrections)
    print "       Average of accuracies obtained is :", numpy.average(accuracies), " and the Standard Deviation is ", \
        numpy.std(accuracies)
    return 0

if __name__ == '__main__':
    main()
