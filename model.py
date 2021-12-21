import enum
import sys
from numpy.core.numeric import outer
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def S1(k, i, x):
    low = max(i - k, 0)
    high = min(len(x) - 1, i + k)

    n1 = 0 if low == 0 else (x[i] - x[i - 1] if low == 1 else max(map(lambda xk: x[i] - xk, x[low:i])))
    n2 = 0 if high == i else (x[i] - x[i + 1] if high == len(x) - 1 else max(map(lambda xk: x[i] - xk, x[i+1:high + 1])))

    return (n1 + n2) / 2


def S2(k, i, x):
    low = max(i - k, 0)
    high = min(len(x), i + k)
    n1 = sum(map(lambda xk: x[i] - xk), x[low:i]) / k
    n2 = sum(map(lambda xk: x[i] - xk), x[i+1:high]) / k

    return (n1 + n2) / 2


def S3(k, i, x):
    low = max(i - k, 0)
    high = min(len(x), i + k)
    n1 = x[i] - (sum(x[low:i]) / k)
    n2 = x[i] - (sum(x[i:high]) / k)

    return (n1 + n2) / 2


def detection(h, k, x):
    output = np.zeros((len(x), 1))
    a = np.zeros((len(x), 1))

    for i in range(len(x)):
        a[i] = S1(k, i, x)

    std = np.std(a[a > 0])
    mean = sum(a[a > 0]) / len(a[a > 0])

    for i in range(len(x)):
        if a[i] > 0 and ((a[i] - mean) > (h * std)):
            for j in range(i - k, i):
                if output[j] > x[i]:
                    output[i] = 0
                    break
                else:
                    output[j] = 0
                    output[i] = x[i]

    return output

if __name__ == "__main__":
    ticker, start, end = 'TSLA', '2020-01-01', '2021-01-01'

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    if len(sys.argv) > 2:
        start = sys.argv[2]
    if len(sys.argv) > 3:
        end = sys.argv[3]

    df = yf.download(ticker,
                    start=start,
                    end=end)

    data = df['Close'].to_list()
    output = detection(1, 3, data)

    ax = plt.gca()

    for (x, y) in enumerate(output):
        if y != 0:
            circle = plt.Circle((x, y), 2, color=cm.jet(0.8), fill=False)
            ax.add_artist(circle)


    plt.plot(data)
    plt.show()