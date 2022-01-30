import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plotLinearReg(slope, intercept):
    plt.subplot(111)
    plt.xlim(0, 400)
    plt.ylim(0, 700)
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="green")

def plotAllPoints(data_set):
    plt.subplot(111)
    x_dim = np.empty((335, 1), float)
    y_dim = np.empty((335, 1), float)
    for i in range(335):
        x_dim[i] = i
        y_dim[i] = data_set[i][1]

    plt.xlim(0, 400)
    plt.ylim(0, 700)
    plt.grid()
    plt.plot(x_dim, y_dim, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")


def PolyCoefficients(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

if __name__ == "__main__":
    data_set = np.empty((335, 1), float)
    matrix_csv = (pd.read_csv("covid_cases.csv"))
    data_set = matrix_csv.to_numpy()
    # print("x:"+str(data_set[0][0])+"   y :"+str(data_set[0][1]))
    # print(data_set)

    # initialize A
    A = np.empty((335, 2), float)
    y = np.empty((335, 1), float)
    for x in range(335):
        A[x][0] = 1
        A[x][1] = x
        y[x] = data_set[x][1]

    #     x = inverse(At * A) * At * y
    #     and then calculate the slope and intercept of estimated line
    inverse_ATranspose_dot_A = np.linalg.inv(np.dot(A.T, A))
    p1 = np.dot(inverse_ATranspose_dot_A, A.T)
    # a0, a1
    x = np.dot(p1, y)
    intercept = x[0]
    slope = x[1]
    # print("xxxxxx")
    # print(x)
    plotAllPoints(data_set)
    plotLinearReg(slope, intercept)
    plt.show()

    print("tests for linear estimate ------->>>>>>")
    # Test 5 sample
    for x in range(330, 335):
        estimated_y = intercept + slope * x
        real_y = data_set[x][1]
        error_y = real_y - estimated_y
        print("for x = "+ str(x) + "we have:")
        print("estimated_y is  : "+ str(estimated_y))
        print("real_y is : "+ str(real_y))
        print("error_y is : "+ str(error_y)+"\n\n")



    # polynomial regression
    # initialize A
    A = np.empty((335, 3), float)
    y = np.empty((335, 1), float)
    for x in range(335):
        A[x][0] = 1
        A[x][1] = x
        A[x][2] = x ** 2
        y[x] = data_set[x][1]

    #     x = inverse(At * A) * At * y
    inverse_ATranspose_dot_A = np.linalg.inv(np.dot(A.T, A))
    p1 = np.dot(inverse_ATranspose_dot_A, A.T)
    # a0, a1, a2
    coeffs = np.dot(p1, y)

    plotAllPoints(data_set)
    x = np.array(range(1, 335))
    plt.subplot(111)
    plt.plot(x, PolyCoefficients(x, coeffs))
    plt.show()


    print("tests for polynomial estimate ------->>>>>>")
    # Test 5 sample
    for x in range(330, 335):
        estimated_y = PolyCoefficients(x, coeffs)
        real_y = data_set[x][1]
        error_y = real_y - estimated_y
        print("for x = "+ str(x) + "we have:")
        print("estimated_y is  : "+ str(estimated_y))
        print("real_y is : "+ str(real_y))
        print("error_y is : "+ str(error_y)+"\n\n")
