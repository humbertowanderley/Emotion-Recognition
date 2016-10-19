from scipy import *
from scipy.linalg import norm, pinv
from sklearn.svm import SVC
from sklearn.externals import joblib
import sklearn.gaussian_process as gpml
import numpy as np

X = []
Y = []

def switch(argument):
    switcher = {
        0: "neutro",
        1: "feliz",
        2: "triste",
        3: "surpresa",
        4: "raiva"
    }
    return switcher.get(argument, "nothing")

if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------



    #file manager
    file = open("features.xls", "r");
    table = file.readlines()

    del table[0]
    for line in table:
    	aux = line.split()

    	valuesFloat = []
    	for valuesStr in aux[0:len(aux)-1]:
    		valuesFloat.append(float(valuesStr))
    	X.append(valuesFloat)
    	Y.append(switch(int(aux[len(aux)-1])))
    file.close()

    X = np.asarray(X)
    Y = np.asarray(Y)


    clf = SVC(kernel='linear',probability=True, tol=1e-3)
    clf.fit(X, Y)

    joblib.dump(clf, 'nn/net.nn')
