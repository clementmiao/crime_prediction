# CS121 Linear regression assignment
# Clement Miao
#
# Generate text answers
#
import sys
from model import *


# useful defined constants for city data
COMPLAINT_COLS = range(0,7)
BURGLARY_COL = 9
CRIME_TOTAL_COL = 19

# useful defined constants for the stock data
STOCKS_IN_DJIA=range(0,30)
STOCKS_NOT_IN_DJIA=range(30,44)
DJIA=44
S_AND_P_500=45

if __name__ == "__main__":
    pass

(col_names, data) = read_file("data/city/training.csv")

#task 2a
print "Task 2a"
print "------------"
for i in COMPLAINT_COLS:
    x_s = data[:, i].reshape(len(data[:,i]), 1)
    y_s = data[:, CRIME_TOTAL_COL]
    beta = linear_regression(x_s ,y_s)
    print col_names[i].lower() + "'s R2: " + str(compute_R2(beta, x_s, y_s))

#task 2b
print "Task 2b"
print "-------------"
x_s = column_stack([data[:, i] for i in COMPLAINT_COLS])
y_s = data[:, CRIME_TOTAL_COL]
beta = linear_regression(x_s, y_s)
print "R2: " + str(compute_R2(beta, x_s, y_s))

#task 3
print "Task 3"
print "-------------"
best_model = compute_best_bivariate(data, COMPLAINT_COLS, CRIME_TOTAL_COL)
print "The best two complaint variables that yield the best bivariate model for predicting total crime are " + col_names[best_model["predictor_var_indices"][0]].lower() + " and " + col_names[best_model["predictor_var_indices"][1]].lower() + " with an R2 of " + str(best_model["R2"]) + "."

#task 4b
print "Task 4b"
print "-------------"
print discover_best_model_with_threshold(data, COMPLAINT_COLS,
                                       CRIME_TOTAL_COL, 0.01)