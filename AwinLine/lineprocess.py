import os
import shutil
from os.path import isfile, isdir, join
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
from PIL import Image

import txaio
txaio.use_twisted()
import argparse
import cv2
import imagehash
import json
import numpy as np
import StringIO
import urllib
import base64
import httplib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import openface
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
import linedb
import lineclass


def test():
    '''
    indentity = linedb.getUserMaxIdentity()
    isExsist = linedb.judmentUserIsExsist("marsh")
    print(indentity)
    print(isExsist)
    indentity = linedb.getUserIdentity("marsh")
    print(indentity)
    '''


def UpdateUser(name, hash, content, representation):
    isExsist = linedb.judmentUserIsExsist(name)
    indentity = linedb.getUserMaxIdentity()
    if(isExsist):
        indentity = linedb.getUserIdentity(name)
        linedb.addTrainingUserPhoto(indentity, hash, ','.join(
            content), ','.join(map(str, representation)))
    else:
        linedb.addTrainingUser(indentity, name)
        linedb.addTrainingUserPhoto(indentity, hash, ','.join(
            content), ','.join(map(str, representation)))


def getKnownUserFromDir():
    for dirname in os.listdir(fileDir + "/known"):
        fullpath = join(fileDir + "/known", dirname)
        if isfile(fullpath):
            continue

        indentity = linedb.getUserIdentity(dirname)
        personalImage = []

        print(indentity)

        for file in os.listdir(fullpath):
            fp = open(os.path.join(fullpath, file), "rb")
            img = fp.read()
            fp.close()
            singleImages = getIamgeInfo(
                "data:image/jpeg;base64," + base64.b64encode(img), indentity)
            if singleImages is None:
                continue
            personalImage.append(singleImages)

        if len(personalImage) > 0:
            for d in personalImage:
                linedb.addKnownUser(indentity, dirname, ','.join(
                    map(str, d["representation"])))

        shutil.rmtree(os.path.join(fileDir + "/known", dirname))


def getUnknowUserFromDir():
    for dirname in os.listdir(fileDir + "/unknown"):
        fullpath = join(fileDir + "/unknown", dirname)
        if isfile(fullpath):
            continue

        print(dirname)

        for file in os.listdir(fullpath):
            fp = open(os.path.join(fullpath, file), "rb")
            img = fp.read()
            fp.close()
            singleImages = getIamgeInfo(
                "data:image/jpeg;base64," + base64.b64encode(img), -1)
            if singleImages is None:
                continue

            linedb.addUnknownUser(dirname, ','.join(
                map(str, singleImages["representation"])))

        # shutil.rmtree(os.path.join(fileDir + "/unknown", dirname))


def trainingUserFromDir():
    for dirname in os.listdir(fileDir + "/images"):
        fullpath = join(fileDir + "/images", dirname)
        if isfile(fullpath):
            continue

        isExsist = linedb.judmentUserIsExsist(dirname)
        indentity = linedb.getUserMaxIdentity()
        if(isExsist):
            indentity = linedb.getUserIdentity(dirname)
        personalImage = []

        print(indentity)

        for file in os.listdir(fullpath):
            fp = open(os.path.join(fullpath, file), "rb")
            img = fp.read()
            fp.close()
            singleImages = getIamgeInfo(
                "data:image/jpeg;base64," + base64.b64encode(img), indentity)
            if singleImages is None:
                continue
            personalImage.append(singleImages)

        if len(personalImage) > 0:
            if(not isExsist):
                linedb.addTrainingUser(indentity, dirname)

            for d in personalImage:
                linedb.addTrainingUserPhoto(indentity, d["hash"], ','.join(
                    d["content"]), ','.join(map(str, d["representation"])))

        shutil.rmtree(os.path.join(fileDir + "/images", dirname))


def getIamgeInfo(dataURL, identity):
    head = "data:image/jpeg;base64,"
    assert(dataURL.startswith(head))
    imgdata = base64.b64decode(dataURL[len(head):])
    imgF = StringIO.StringIO()
    imgF.write(imgdata)
    imgF.seek(0)
    img = Image.open(imgF)
    buf = np.fliplr(np.asarray(img))
    rgbFrame = np.zeros((1024, 1024, 3), dtype=np.uint8)
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
                                  landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        if alignedFace is None:
            continue
        phash = str(imagehash.phash(Image.fromarray(alignedFace)))
        rep = net.forward(alignedFace)
        content = [str(x) for x in alignedFace.flatten()]
        singleImages = {
            "hash": phash,
            "identity": identity,
            "content": content,
            "representation": rep.tolist()
        }
        return singleImages


def getTrainingUser():
    return linedb.getTrainingUser()


def getTrainingUserData():
    return linedb.getTrainingUserData()


def addRecognizeImage(imageDescription, imageBase64, users):
    linedb.addRecognizeImage(imageDescription, imageBase64, users)


def verificationPredict(identity, decision, peopleLen):
    capture = []
    targetPosition = 0
    for x in xrange(0, identity):
        targetPosition += (peopleLen - 1) - x

    for x in xrange(0, (peopleLen - 1 - identity)):
        capture.append(decision[targetPosition + x])

    idx = 0
    if(len(capture) < peopleLen - 1):
        for x in xrange(0, identity):
            targetDistance = identity - x - 1
            capture.append(decision[idx + targetDistance])
            idx += (peopleLen - 1) - x

    print("-----------------------------------")
    print("identity:" + str(identity))
    print(decision)
    print(capture)
    print("-----------------------------------")

    standard = 0.4
    judment = True
    for x in xrange(0, len(capture)):
        if(abs(capture[x]) < standard):
            judment = False
            break
    if(not judment):
        return -1

    for x in xrange(0, len(capture)):
        if(abs(capture[x]) > 0.6):
            judment = True
            break
        else:
            judment = False
    if(not judment):
        return -1

    for x in xrange(0, len(capture)):
        if (abs(capture[x]) > 0.8):
            judment = True
            break

    if(judment):
        return identity

    '''
    print("identity:" + str(identity))
    '''
    return identity


def verificationPredictForSVM2(identity, decision, peopleLen):
    threshold = [[0.089886331, 0.6110], [0.411147261, 0.6632], [0.90324, 1.0406],
                 [0.0, 0.0], [0.9995, 1.133], [0.9995, 1.034]]
    capture = []
    targetPosition = 0
    for x in xrange(0, identity):
        targetPosition += (peopleLen - 1) - x

    for x in xrange(0, (peopleLen - 1 - identity)):
        capture.append(decision[targetPosition + x])

    idx = 0
    if(len(capture) < peopleLen - 1):
        for x in xrange(0, identity):
            targetDistance = identity - x - 1
            capture.append(decision[idx + targetDistance])
            idx += (peopleLen - 1) - x

    print("-----------------------------------")
    print("identity:" + str(identity))
    print(decision)
    print(capture)
    print("threshold[identity][0]:" + str(threshold[identity][0]))
    print("threshold[identity][1]:" + str(threshold[identity][1]))
    print("-----------------------------------")

    standard = threshold[identity][0]
    judment = True
    for x in xrange(0, len(capture)):
        if(abs(capture[x]) < standard):
            judment = False
            break
    if(not judment):
        return -1

    for x in xrange(0, len(capture)):
        if (abs(capture[x]) > threshold[identity][1]):
            judment = True
            break
        else:
            judment = False
    if(not judment):
        return -1

    if(judment):
        return identity

    return identity


def verificationPredictForAdaboost(identity, decision):
    print("----------------------------------")
    print(identity)
    print("----------------------------------")
    print(decision[identity])
    print("----------------------------------")
    print(decision)
    print("----------------------------------")
    if(decision[identity] < 53):
        identity = -1
    return identity


def verificationPredictForAdaboost2(identity, decision):
    threshold = [23.0168226681, 27.8255271306,
                 14.7640708712, 4.3530566086, 78.995, 78.685]
    print("----------------------------------")
    print(identity)
    print("----------------------------------")
    print(decision[identity])
    print("----------------------------------")
    print(decision)
    print("----------------------------------")
    if(decision[identity] < threshold[identity]):
        identity = -1
    return identity


def verificationPredictForRandom(identity, predictProba):
    print("----------------------------------")
    print(identity)
    print("----------------------------------")
    print(predictProba[identity])
    print("----------------------------------")
    print(predictProba)
    print("----------------------------------")
    if(predictProba[identity] < 0.28):
        identity = -1
    return identity


def verificationPredictForRandom2(identity, predictProba):
    threshold = [0.4125, 0.575, 0.475, 0.3375, 0.815, 0.86]
    print("----------------------------------")
    print(identity)
    print("----------------------------------")
    print(predictProba[identity])
    print("----------------------------------")
    print(predictProba)
    print("----------------------------------")
    if(predictProba[identity] < threshold[identity]):
        identity = -1
    return identity


def getDecisionPredict(identity, decision, peopleLen):
    capture = []
    pre = []

    targetPosition = 0
    for x in xrange(0, identity):
        targetPosition += (peopleLen - 1) - x

    for x in xrange(0, (peopleLen - 1 - identity)):
        capture.append(decision[targetPosition + x])
        pre.append(targetPosition + x)

    idx = 0
    if(len(capture) < peopleLen - 1):
        for x in xrange(0, identity):
            targetDistance = identity - x - 1
            pre.append(idx + targetDistance)
            idx += (peopleLen - 1) - x

    return pre


def queryRecognizeImage(keyWord):
    return linedb.queryRecognizeImage(keyWord)


def getQueryImage(messageId, imageSeq):
    imageBase = linedb.getQueryImage(imageSeq.replace("#", ""))
    newImageOriginal = Image.new('RGB', (1500, 1500))
    newImageOriginal.save(os.path.join(fileDir, "wb") +
                          "/" + messageId + ".jpg", "JPEG")

    imgData = base64.b64decode(imageBase)
    leniyimg = open(os.path.join(fileDir, "wb") +
                    "/" + messageId + ".jpg", 'wb')
    leniyimg.write(imgData)
    leniyimg.close()


def getTestKnownData():
    return linedb.getTestKnownData()


def getTestKnownPredictData():
    return linedb.getTestKnownPredictData()


def getTestUnknownData():
    return linedb.getTestUnknownData()


def updateTestUnknownData(dataSeq, predictName):
    linedb.updateTestUnknownData(dataSeq, predictName)


def updateTestknownData(dataSeq, predictName):
    linedb.updateTestknownData(dataSeq, predictName)


def NormalizationImage(dataURL):
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
    bb = align.getLargestFaceBoundingBox(rgbFrame)
    bbs = [bb] if bb is not None else []
    for bb in bbs:
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                  landmarks=landmarks,
                                  landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        if alignedFace is None:
            continue
        return alignedFace


def ProcessMultiImage(folderName):
    for fileName in os.listdir(fileDir + "/" + folderName):
        fullpath = fileDir + "/" + folderName
        fp = open(os.path.join(fullpath, fileName), "rb")
        img = fp.read()
        fp.close()
        try:
            NormalizationMultiImage(
                "data:image/jpeg;base64," + base64.b64encode(img), fullpath, fileName)
        except Exception as e:
            print(e)
            continue


def NormalizationMultiImage(dataURL, fullpath, fileName):
    head = "data:image/jpeg;base64,"
    assert(dataURL.startswith(head))
    imgdata = base64.b64decode(dataURL[len(head):])
    imgF = StringIO.StringIO()
    imgF.write(imgdata)
    imgF.seek(0)
    img = Image.open(imgF)
    buf = np.fliplr(np.asarray(img))
    rgbFrame = np.zeros((1024, 1280, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = buf[:, :, 2]
    rgbFrame[:, :, 1] = buf[:, :, 1]
    rgbFrame[:, :, 2] = buf[:, :, 0]
    bbs = align.getAllFaceBoundingBoxes(rgbFrame)
    if not bbs:
        print("Unable to find a face: {}".format(rgbFrame))
    i = 0
    for bb in bbs:
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                  landmarks=landmarks,
                                  landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        if alignedFace is None:
            continue
        newFileName = fileName.split('.')
        img = Image.fromarray(alignedFace, 'RGB')
        img.save(os.path.join(fullpath, newFileName[
                 0] + '_' + str(i) + '.' + newFileName[1]))
        i += 1
