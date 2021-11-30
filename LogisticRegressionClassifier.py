import numpy as np
import math
from math import e
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# turn visualize off if number of training examples does not equal two
Visualize = True


# visualize the data by constructing a colored scatterplot
# for now only works with n = 2 training examples so turn off if n changes
def visualizeData(y, x, p, w, isActual):
    fig = plt.figure(figsize=(7, 7))
    plt.style.use('seaborn-paper')
    ax = plt.axes(projection='3d')

    # color each point
    for c in range(len(p)):
        if isActual:
            if y[c] == 1:
                ax.scatter3D(x[1][c], x[2][c], p[c], color='r')
            else:
                ax.scatter3D(x[1][c], x[2][c], p[c], color='b')
        else:
            if round(p[c]) == 1:
                ax.scatter3D(x[1][c], x[2][c], p[c], color='r')
            else:
                ax.scatter3D(x[1][c], x[2][c], p[c], color='b')

    if isActual:
        plt.title("Actual Labels")
    else:
        plt.title("Predicted Labels")

    # set titles, labels, and view angle
    ax.view_init(2, 280)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("P-Value")


# displays the confusion matrix using predictions and actual labels
def displayConfusionMatrix(y, p):
    # turn the p values into actual predictions
    predictions = []
    for i in range(len(p)):
        predictions.append(round(p[i]))

    # make confusion matrix
    cm = confusion_matrix(y, predictions)
    ax = sns.heatmap(cm, annot=True, cmap="PuBuGn")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


# graphs the training curve after execution
def graphTrainingCurve(nWU, pCorrect):
    plt.figure(figsize=(7, 7))
    plt.style.use('seaborn-paper')

    plt.plot(nWU, pCorrect)
    plt.title("Training Curve")
    plt.xlabel("Number of Weight Updates")
    plt.ylabel("Proportion Correct")


# calculates p-values for predictions
def predict(x, w):
    # use the logistic function to compute probabilities for each example
    pValues = []
    for i in range(len(x[0])):
        eq = 0
        # add each weight times its corresponding example instance
        for j in range(len(w)):
            eq += w[j] * x[j][i]

        pValues.append(1 / (1 + e ** -eq))

    return pValues


# counts the number false positives and false negatives, as well as correct predictions for specificity/sensitivity
def numError(p, actual):
    # class 0 and class 1 error counters
    error0, error1 = 0, 0
    # correct prediction count
    negCorrect, posCorrect = 0, 0
    for i in range(len(actual)):
        # round the p values to see what their most likely value
        pred = round(p[i])
        # if class 0 error increment count0, else if class 1 error increment count1
        if pred == 1 and actual[i] == 0:
            error0 += 1
        elif pred == 0 and actual[i] == 1:
            error1 += 1
        elif pred == 0 and actual[i] == 0:
            negCorrect += 1
        elif pred == 1 and actual[i] == 1:
            posCorrect += 1

    return error0, error1, negCorrect, posCorrect


# calculates the loglikelihood given some p-values
def logLikelihood(p, y, w):
    # add up the log odds of each probability depending on their label
    logOdds = []

    for i in range(len(y)):
        if y[i] == 1:
            logOdds.append(math.log(p[i], e))
        else:
            logOdds.append(math.log(1 - p[i], e))

    return sum(logOdds)


# uses gradient descent to find the best fitting line
def gradientDescent(x, w, p, actual, a):
    size = len(actual)
    losses = []

    # for each weight
    for i in range(len(w)):
        # calculate loss for corresponding example
        loss = 0
        for j in range(size):
            loss += (p[j] - actual[j]) * x[i][j]
        losses.append(loss)

        # update corresponding weight
        w[i] -= loss * a / size

    # check if all weights converged
    for l in losses:
        if round(l, 1) != 0:
            return False
    return True


def logisticRegression():
    # labels
    Y = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    # training examples
    X = [[], [12, 15, 9, 9, 10, 7, 16, 8, 15, 6, 10, 16, 14, 20, 12],
             [39, 38, 26, 20, 30, 22, 10, 37, 38, 16, 22, 36, 28, 34, 35]]

    for y in Y:
        # w0 is always multiplied by x0 which always equals 1
        X[0].append(1)

    # initialize weights
    weights = []
    for x in X:
        weights.append(0.1)
    # counter for weight updates
    count = 0
    numWeightUpdates = []
    class0, class1 = 0, 0
    negc, posc = 0, 0
    scores = []

    # learning rate
    stepSize = 0.01

    # emulating do while loop
    while True:
        # use the logistic function to compute p-values
        pValues = predict(X, weights)

        # see the proportion of correctly predicted values
        class0, class1, negc, posc = numError(pValues, Y)
        scores.append((len(Y) - (class0 + class1)) / len(Y))
        numWeightUpdates.append(count)
        count += len(weights)

        # update the weights using gradient descent, if convergence occurs break
        if gradientDescent(X, weights, pValues, Y, stepSize):
            break

    # display final weights and predictions
    print("Results:\n")
    for i in range(len(Y)):
        for j in range(1, len(X)):
            print("X{}:".format(j), X[j][i])
        print("Predicted Y:", round(pValues[i]), ", Actual Y:", Y[i], '\n')
    print("Final weights:")
    for w in range(len(weights)):
        print("W{}:".format(w), weights[w])

    # display confusion matrix
    displayConfusionMatrix(Y, pValues)

    # Calculate overall error rate, accuracy, false negative rate, false positive rate, specificity, and sensitivity
    overallErrorRate = (class0 + class1) / len(Y)
    accuracy = 1 - overallErrorRate

    falseNegRate = class0 / (class0 + negc)
    falsePosRate = class1 / (class1 + posc)
    specificity = 1 - falseNegRate
    sensitivity = 1 - falsePosRate

    print("\nAccuracy:", round(accuracy, 2), "Specificity:", round(specificity, 2), "Sensitivity:", round(sensitivity, 2))

    # Calculate R squared and the p-value of our model
    # find log likelihood of fittest weights
    LLFit = logLikelihood(pValues, Y, weights)

    # find the number of true labels divided by total and fill a list with them
    posProbs = []
    for y in Y:
        posProbs.append(sum(Y)/len(Y))
    # and find the log likelihood of positive probability list
    LLOverall = logLikelihood(posProbs, Y, weights)

    # R squared value gives us the strength of the relationships
    RR = (LLOverall - LLFit) / LLOverall

    # calculate chi square value and degrees of freedom to obtain p value
    chiSquare = 2 * (LLFit - LLOverall)
    df = len(weights) - 1
    # p value tells us if the relationship is significantly different from random noise
    p = 1 - stats.chi2.cdf(chiSquare, df)

    print("\nR-Squared:", round(RR, 2), "p-value:",
          round(p, 2))

    # display training curve
    graphTrainingCurve(numWeightUpdates, scores)

    # get a feel for the data by visualizing them
    # for number of training examples not equal to two, turn visualize off
    if Visualize:
        visualizeData(Y, X, pValues, weights, True)
        visualizeData(Y, X, pValues, weights, False)

    plt.show()


logisticRegression()
