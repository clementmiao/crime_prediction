# Linear regression assignment
# Clement Miao
#
# 
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

#task 1
for i in range(3):
    x_s = data[:, i]
    y_s = data[:, CRIME_TOTAL_COL]
    beta = linear_regression(x_s.reshape(len(x_s), 1) ,y_s)
    regs = apply_beta(beta,x_s.reshape(len(x_s), 1))
    figure(i+1)
    scatter(x_s, y_s)
    plot(x_s, regs)
    xlabel(col_names[i].lower())
    ylabel(col_names[CRIME_TOTAL_COL].lower())
    title("Task 1: " + col_names[i].lower() + " vs. " + col_names[CRIME_TOTAL_COL].lower())
    savefig("pix/task_1_" + col_names[i].lower() + ".png")
    close('all')

#task 4a
x_s = COMPLAINT_COLS
y_s = []
for k in x_s:
    best_model = discover_best_k_model((k+1), data, COMPLAINT_COLS, CRIME_TOTAL_COL)
    y_s.append(best_model["R2"])
figure()
(x_s).insert(0,-1)
x_s = array((x_s)) + 1
y_s.insert(0,0) 
plot(x_s, y_s)
xlabel('number of variables, K')
ylabel('R2')
title('Task 4a: K versus R2 value for the best K-variable model for predicting total crime')
savefig("pix/task_4_k_variable.png")
close('all')



