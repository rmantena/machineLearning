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
    new_array = numpy.zeros((len(matrix),149))
    for count in range(len(matrix)):
        for a in range(len(matrix[count])):
            new_array[count][a] = matrix[count][a]
            # new_array[count][a] = int(''.join(matrix[count][a]))
    return new_array

# Winnow Algorithm Code
def winnow(data):
    w = numpy.ones(146)                             # Initializing the Weights vector
    theta = 147                                     # Initializing theta
    corrections = 0
    for count in range(len(data)):                  # For each record
        data_here = data[count]                     # Copy 2D array into a 1D array
        y_here = data_here[148]                     # Separate label,
        x_here = data_here[1:147]                   # Separate features,
        weekday = data_here[0]                      # Separate the Weekday boolean. (Not using this as a feature yet)
        dot_prod = numpy.dot(w,x_here)
        if weekday:                                 # Ignore Weekends
            if dot_prod - theta > 0:                # Winnow update code
                y_new = 1
            else:
                y_new = 0
            if y_here == 1 and y_new ==0:           # If mistake is made
                corrections += 1
                for mycount in range(len(w)):       # Loop over all features to find the True features
                    if x_here[mycount] == 1:        # If feature is 1
                        w[mycount] = 2*w[mycount]   # Promotive update
            elif y_here == 0 and y_new == 1:        # If mistake is made
                corrections += 1
                for mycount in range(len(w)):       # Loop over features
                    if x_here[mycount] == 1:        # If feature is 1
                        w[mycount] = w[mycount]/2   # Demotive update
    return w,corrections

# Balanced Winnow Algorithm Code
def balanced_winnow(data):
    w_pos = numpy.ones(146)                         # Initializing the Weights vector
    w_neg = numpy.ones(146)                         # Initializing the Weights vector
    theta = 147                                     # Initializing theta
    corrections = 0
    for count in range(len(data)):                  # For each record
        data_here = data[count]                     # Copy 2D array into a 1D array
        y_here = data_here[148]                     # Separate label,
        x_here = data_here[1:147]                   # Separate features,
        weekday = data_here[0]                      # Separate the Weekday boolean. (Not using this as a feature yet)
        w = w_pos-w_neg
        dot_prod = numpy.dot(w,x_here)
        if weekday:                                 # Ignore Weekends
            if dot_prod - theta > 0:                # Winnow update code
                y_new = 1
            else:
                y_new = 0
            if y_here == 1 and y_new ==0:           # If mistake is made
                corrections += 1
                for mycount in range(len(w_pos)):       # Loop over all features to find the True features
                    if x_here[mycount] == 1:        # If feature is 1
                        w_pos[mycount] = 2*w_pos[mycount]   # Promotive update
                        w_neg[mycount] = w_neg[mycount]/2
            elif y_here == 0 and y_new == 1:        # If mistake is made
                corrections += 1
                for mycount in range(len(w_pos)):       # Loop over features
                    if x_here[mycount] == 1:        # If feature is 1
                        w_pos[mycount] = w_pos[mycount]/2   # Demotive update
                        w_neg[mycount] = 2*w_neg[mycount]
    w = w_pos - w_neg
    return w,corrections


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
            predicted = numpy.dot(final_vector,x_here)  # What is the prediction of W (final vector) ?
            if predicted >= theta:
                y = 1
            else:
                y = 0
            print "Predicted : ", y, " Actual : ", y_here
            if predicted >= theta and y_here == 1:      # Does it match the label ?
                print "1"
                correct_predictions += 1                # Increment correct_predictions
            elif predicted < theta and y_here == 0:     # Same as above
                print "2"
                correct_predictions += 1                # Increment correct_predictions
            total_predictions_weekdays += 1
    print "correct_predictions : ", correct_predictions, "total_predictions_weekdays : ", total_predictions_weekdays
    correct_predictions = (correct_predictions/float(total_predictions_weekdays))*100    # Convert to average
    return correct_predictions                          # Return predictions.


# Main function
def main():
    learning_rate = 1                                   # Initializing learning rate
    # Variable to define the algorithm to run
    # Select 1 for Winnow
    # Select 2 for Balanced Winnow
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
            (final_vec, correction) = winnow(array)
            curr_acc = validate_winnow(final_vec,test_array)
        elif (Algorithm_Select == 2):
            (final_vec, correction) = balanced_winnow(array)
            curr_acc = validate_winnow(final_vec,test_array)
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
