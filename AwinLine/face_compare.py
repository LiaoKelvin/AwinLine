import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import argparse
import cv2
import imagehash
import json
import StringIO
import urllib
import base64

from PIL import Image

import txaio
txaio.use_twisted()

import numpy as np
from sklearn.grid_search import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import openface

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


people = []
images = {}
if args.unknown:
    self.unknownImgs = np.load("./examples/web/unknown.npy")


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


def getAutoTrainingFilePath():
    trainingPeople = []
    trainingImages = []
    allFiles = []
    print("--------------getAutoTrainingFilePath---------------")
    for root, dirs, files in os.walk(fileDir + "/images"):
        for d in dirs:
            trainingPeople.append(d)
            people.append(d)
            print(trainingPeople)

        for f in files:
            ide = 0
            for p in trainingPeople:
                if p in f:
                    ide = trainingPeople.index(p)
                    break

            fp = open(os.path.join(root, f), "rb")
            img = fp.read()
            fp.close()
            print(f)
            print(ide)
            singleImages = processFrameForAutoTraining(
                "data:image/jpeg;base64," + base64.b64encode(img), ide)
            if singleImages is None:
                print(singleImages)
                continue
            trainingImages.append(singleImages)
            images[singleImages["hash"]] = Face(
                np.array(singleImages["representation"]), singleImages["identity"])

    print("--------------getAutoTrainingFilePath----End--------")


def processFrameForAutoTraining(dataURL, identity):
    head = "data:image/jpeg;base64,"
    assert(dataURL.startswith(head))
    imgdata = base64.b64decode(dataURL[len(head):])
    imgF = StringIO.StringIO()
    imgF.write(imgdata)
    imgF.seek(0)
    img = Image.open(imgF)
    buf = np.fliplr(np.asarray(img))
    rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]

    identities = []
    bb = align.getLargestFaceBoundingBox(rgbFrame)
    bbs = [bb] if bb is not None else []
    for bb in bbs:
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                  landmarks=landmarks,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            continue
        phash = str(imagehash.phash(Image.fromarray(alignedFace)))
        if phash in images:
            identity = images[phash].identity
        else:
            rep = net.forward(alignedFace)
            images[phash] = Face(rep, identity)
            content = [str(x) for x in alignedFace.flatten()]
            singleImages = {
                "hash": phash,
                "identity": identity,
                "content": content,
                "representation": rep.tolist()
            }
            return singleImages


def getData():
    X = []
    y = []
    for img in images.values():
        X.append(img.rep)
        y.append(img.identity)

    numIdentities = len(set(y + [-1])) - 1
    if numIdentities == 0:
        return None

    if args.unknown:
        numUnknown = y.count(-1)
        numIdentified = len(y) - numUnknown
        numUnknownAdd = (numIdentified / numIdentities) - numUnknown
        if numUnknownAdd > 0:
            print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
            for rep in unknownImgs[:numUnknownAdd]:
                    # print(rep)
                X.append(rep)
                y.append(-1)

    X = np.vstack(X)
    y = np.array(y)
    return (X, y)


def compare():

    getAutoTrainingFilePath()

    '''
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]
    '''
    names = ["Linear SVM", "Quadratic Discriminant Analysis"]
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']}
    ]
    classifiers = [
        # KNeighborsClassifier(3),
        GridSearchCV(SVC(kernel="linear", C=0.025), param_grid, cv=5),
        #GridSearchCV(SVC(gamma=2, C=1), param_grid, cv=5),
        # DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    figure = plt.figure(figsize=(27, 9))
    (X, y) = getData()

    X = StandardScaler().fit_transform(X)

    X_train = X
    X_test = X

    y_train = y
    y_test = y
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, len(classifiers) + 1, 1)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i = 1
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(1, len(classifiers) + 1, 1 + i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = clf.decision_function(X_test)
        else:
            #Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = clf.predict_proba(X_test)[:, 1]
        #print(Z)
        # Put the result into a color plot
        #Z = Z.reshape(xx.shape)
        #ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        #ax.contourf(X_train[:, 0], X_train[:, 1], Z, cmap=cm, alpha=.8)
        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(x_max - .3, y_min + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

    figure.subplots_adjust(left=.02, right=.98)
    plt.show()
    print("-----------show------------")

compare()
