import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def derivative(f, x, h):
    return [
        f(i + h) - f(i) / h for i in x
    ]

def main():
    
    dist = np.arange(0, 110, 10)
    time = np.array([
        0, 1.85, 2.87, 3.78, 
        4.65, 5.50, 6.32, 7.14,
        7.96, 8.79, 9.69])

    # overfitting the function with deg=3
    coefs = np.polyfit(time, dist, deg=3)
    func = np.poly1d(coefs)
    output = func(time)
    dy_dx = derivative(func, time, 1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].scatter(time, dist)
    ax[0].plot(time, output, color='black')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('distance (meters)')

    ax[1].plot(time, dy_dx)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('velocity')
    
    plt.show()


if __name__ == "__main__":
    main()