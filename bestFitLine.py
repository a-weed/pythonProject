from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def createDataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def bestFitSlopeAndIntercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
        ((mean(xs)**2) - (mean(xs**2))))
    b = mean(ys) - m*mean(xs)
    return m, b

def squaredError(ysOrig, ysLine):
    return sum((ysLine-ysOrig)**2)

def coefficientOfDetermination(ysOrig, ysLine):
    yMeanLine = [mean(ysOrig) for y in ysOrig]
    squaredErrorRegr = squaredError(ysOrig, ysLine)
    squaredErrorYMean = squaredError(ysOrig, yMeanLine)
    return 1 - (squaredErrorRegr/squaredErrorYMean)

xs, ys = createDataset(40,80,2,correlation=False)


m, b = bestFitSlopeAndIntercept(xs,ys)

regressionLine = [(m*x) + b for x in xs]

predictX = 8
predictY = (m*predictX + b)

rSquared = coefficientOfDetermination(ys, regressionLine)
print(rSquared)

plt.scatter(xs,ys)
plt.scatter(predictX, predictY, color = 'g')
plt.plot(xs, regressionLine)
plt.show()
