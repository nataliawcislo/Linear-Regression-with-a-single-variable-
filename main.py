import matplotlib.pyplot as plt
import numpy as np


def wczytaj_dane():
    with open("dane.txt") as f:
        x = []
        y = []
        file = f.readlines()
        for i in range(0, len(file)):
            plik_temp = list(file[i].split())
            if (i != 0):
                x.append(float(plik_temp[0]))
                y.append(float(plik_temp[1]))
    return (x, y)


def scalling(x, y):
    print("Mean of x values is %f and median is %f\n" % (np.mean(x), np.median(x)))
    print("Mean of y values is %f and median is %f\n" % (np.mean(y), np.median(y)))


def feature_data(x, y):
    plt.plot_date(x, y, marker='*', color='deeppink', label="data")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("traning data")
    plt.legend()
    plt.show()


def hypothesis(theta_array, x):
    return theta_array[1] * x + theta_array[0]


def cost_function(theta_array, x, y, m):
    J = 0
    for i in range(m):
        J += ((theta_array[0] + theta_array[1] * x[i]) - y[i]) ** 2
    return J / 2 * m


def i_thetas(theta_array, x, y, alpha, m):
    sum_0 = 0
    sum_1 = 0

    for i in range(m):
        sum_0 += (theta_array[0] + theta_array[1] * x[i]) - y[i]
        sum_1 += x[i] * ((theta_array[0] + theta_array[1] * x[i]) - y[i])
    new_theta_0 = theta_array[0] - alpha * (sum_0) / m
    new_theta_1 = theta_array[1] - alpha * (sum_1) / m
    updated_theta_array = [new_theta_0, new_theta_1]
    return updated_theta_array


def gradient(x, y, alpha, iteration):
    m = len(x)
    theta_0 = 0  # bias
    theta_1 = 0  # weight
    theta_array = [theta_0, theta_1]
    cost_values = []

    for i in range(iteration):
        theta_array = i_thetas(theta_array, x, y, alpha, m)
        cost_values.append(cost_function(theta_array, x, y, m))

    x = np.arange(0, len(cost_values), step=1)
    plt.plot(x, cost_values, label="Cost Function", color='orange')
    plt.title("Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function Value")
    plt.legend()
    plt.show()
    return theta_array


def test(x,t):
    z=[]
    for i in range(len(x)):
        z.append(t[1] * x[i] + t[0])
    plt.plot(x, z, color='red', label="data")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Test")
    plt.legend()
    plt.show()


def main():
    a = 0.01  # defining a learning rate
    i = 15000  # Setting the number of iterations
    x, y = wczytaj_dane()
    feature_data(x, y)
    scalling(x, y)
    t =gradient(x, y, a, i)
    test(x, t)


main()
