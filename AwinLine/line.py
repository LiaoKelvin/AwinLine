#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

from flask import Flask, request, abort, send_file

from flask import json

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage
)

from PIL import Image

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.python import log
from twisted.internet import reactor

import argparse
import cv2
import imagehash
import json
import numpy as np
import os
import StringIO
import urllib
import base64
import httplib
import xlwt
import sklearn.metrics
import cPickle

from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.linear_model import SGDClassifier

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ImageDraw
import ImageFont

import openface
import lineprocess

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

inet_addr = "127.0.0.1"

app = Flask(__name__)

line_bot_api = LineBotApi(
    '2Wls0Ci2mcvzALEDZTIl7l5bvN87TE6cAbftQO9v1kAnGJ6eu8v87HyDEQbaKBXVW29tAYblh5EBzXvBzbJHSso3u1qIQu56VuMduHTjZB9tZ3XKexqSVyRxpGNzX3GbuCe8hpFrUjwtpwDhXXW1ZQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('feeee8de9c08e9d17543375326f4c6b0')

webUrl = "https://787e79e4.ngrok.io"


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@app.route("/print", methods=['GET'])
def getMethod():
    return json.dumps({"status": 200, "comment": "[ Get Method ] Hello World"})


@app.route("/wb/<path:filename>", methods=['GET', 'POST'])
def getImage(filename):
    return send_file(os.path.join(fileDir, "wb") + "/" + filename, mimetype='image/JPG')


@handler.add(MessageEvent)
def handle_message(event):
    global awinIns
    # print(event)
    talkType = str(event.source).split('"type":')[
        1].replace('"', "").replace('}', "").strip()
    # print(talkType)
    userId = ""
    if talkType == "group":
        userId = str(event.source).split('"groupId":')[
            1].replace('"', "").replace('}', "").strip()
    else:
        userId = str(event.source).split('"userId":')[
            1].replace('"', "").replace('}', "").strip()
    dialogue = GetDialogue(userId)

    if event.message.type == "text":
        messageText = event.message.text
        if(len(dialogue) == 2 and dialogue[0] == "#Query"):
            try:
                lineprocess.getQueryImage(event.message.id, messageText)
                image_message = ImageSendMessage(
                    original_content_url=webUrl + '/wb/' + event.message.id + '.jpg',
                    preview_image_url=webUrl + '/wb/' + event.message.id + '.jpg'
                )
                line_bot_api.reply_message(event.reply_token, image_message)
            except Exception as e:
                ClearDialogue(userId)
                print(e)
        else:
            try:
                replyMessage = DialogueProcess(messageText, userId)
                if(replyMessage != ""):
                    line_bot_api.reply_message(
                        event.reply_token, TextSendMessage(text=replyMessage))
            except Exception as e:
                print(e)
        '''
        global ofsp
        if(messageText == "#AutoTrain"):
            awinIns = ""
            ofsp.getAutoTrainingFilePath()
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='Train Is Completed'))
        elif(messageText == "#Incremental"):
            awinIns = ""
            ofsp.incrementalTraining()
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='Incremental Train Is Completed'))
        elif ("#" in messageText):
            awinIns = messageText.replace("#", "")
            if not os.path.exists(os.path.join(fileDir, "images") + "/" + awinIns):
                os.makedirs(os.path.join(fileDir, "images") + "/" + awinIns)
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=event.message.text))
        '''
    elif event.message.type == "image":
        message_content = line_bot_api.get_message_content(event.message.id)
        ImageProcess(message_content.iter_content(), event.message.id, userId)

        if(len(dialogue) == 2 and dialogue[0] == "#Recognize"):
            SetDialogue(event.message.id, userId)
            fp = open(os.path.join(fileDir, "wb") +
                      "/" + event.message.id + ".jpg", 'rb')
            img = fp.read()
            fp.close()
            ofsp.processFrame("data:image/jpeg;base64," +
                              base64.b64encode(img), -1,  event.message.id)
            images = ofsp.getUserNameFromImage(
                "data:image/jpeg;base64," + base64.b64encode(img))
            SetWaittingImages(images, userId)
            image_message = ImageSendMessage(
                original_content_url=webUrl + '/wb/' + event.message.id + '_o.jpg',
                preview_image_url=webUrl + '/wb/' + event.message.id + '_p.jpg'
            )
            line_bot_api.reply_message(event.reply_token, image_message)
        elif(len(dialogue) == 0):
            SetDialogue("#Recognize", userId)
            SetDialogue(" ", userId)
            SetDialogue(event.message.id, userId)
            fp = open(os.path.join(fileDir, "wb") +
                      "/" + event.message.id + ".jpg", 'rb')
            img = fp.read()
            fp.close()
            ofsp.processFrame("data:image/jpeg;base64," +
                              base64.b64encode(img), -1,  event.message.id)
            images = ofsp.getUserNameFromImage(
                "data:image/jpeg;base64," + base64.b64encode(img))
            SetWaittingImages(images, userId)
            '''
            description = AnalyzeAnImage(event.message.id + '_o.jpg')
            putImageText(description, event.message.id + '_o')
            putImageText(description, event.message.id + '_p')
            '''
            image_message = ImageSendMessage(
                original_content_url=webUrl + '/wb/' + event.message.id + '_o.jpg',
                preview_image_url=webUrl + '/wb/' + event.message.id + '_p.jpg'
            )
            # ClearDialogue(userId)
            line_bot_api.reply_message(event.reply_token, image_message)

        '''
        floderPath = "wb"
        filename = ""
        if awinIns.strip() != "":
            filename = awinIns
            floderPath = "images/" + awinIns
        image = Image.new('RGB', (1500, 1500))
        image.save(os.path.join(fileDir, floderPath) + "/" + filename +
                   event.message.id + ".jpg", "JPEG")

        with open(os.path.join(fileDir, floderPath) + "/" + filename + event.message.id + ".jpg", 'wb') as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)

        image = Image.open(os.path.join(fileDir, floderPath) + "/" + filename +
                           event.message.id + ".jpg")
        nim = image.resize((400, 300), Image.BILINEAR)
        image.close()
        os.remove(os.path.join(fileDir, floderPath) + "/" + filename +
                  event.message.id + ".jpg")
        nim.save(os.path.join(fileDir, floderPath) + "/" + filename +
                 event.message.id + ".jpg", "JPEG")

        if awinIns.strip() == "":
            fp = open(os.path.join(fileDir, floderPath) + "/" +
                      event.message.id + ".jpg", 'rb')
            img = fp.read()
            fp.close()

            try:
                print()
                ofsp.processFrame("data:image/jpeg;base64," +
                                  base64.b64encode(img), -1,  event.message.id)
            except:
                print()

            # print(webUrl + '/wb/' + event.message.id + '.jpg')
            description = AnalyzeAnImage(event.message.id + '_o.jpg')
            print(description)
            putImageText(description, event.message.id + '_o')
            putImageText(description, event.message.id + '_p')
            image_message = ImageSendMessage(
                original_content_url=webUrl + '/wb/' + event.message.id + '_od.jpg',
                preview_image_url=webUrl + '/wb/' + event.message.id + '_pd.jpg'
            )
            image_message = ImageSendMessage(
                original_content_url=webUrl + '/wb/' + event.message.id + '_o.jpg',
                preview_image_url=webUrl + '/wb/' + event.message.id + '_p.jpg'
            )
            print(image_message)
            line_bot_api.reply_message(event.reply_token, image_message)
            '''


def DialogueProcess(messageText, userId):
    dialogue = GetDialogue(userId)
    if(messageText == "#CreateNewPerson"):
        SetDialogue(messageText, userId)
        return "請輸入人員名稱"

    elif (len(dialogue) == 1 and dialogue[0] == "#CreateNewPerson"):
        if not os.path.exists(os.path.join(fileDir, "images") + "/" + messageText):
            os.makedirs(os.path.join(fileDir, "images") + "/" + messageText)
        SetDialogue(messageText, userId)
        return u"請上傳 " + messageText + u" 的個人照片，上傳完畢 請輸入:#StartCreate"

    elif (len(dialogue) == 2 and dialogue[0] == "#CreateNewPerson" and messageText == "#StartCreate"):
        lineprocess.trainingUserFromDir()
        ClearDialogue(userId)
        ofsp.setTrainingData(lineprocess.getTrainingUser(),
                             lineprocess.getTrainingUserData())
        return "人員已新增"

    elif (messageText == "#Recognize"):
        SetDialogue(messageText, userId)
        return "請輸入圖片的描述"

    elif(len(dialogue) == 1 and dialogue[0] == "#Recognize"):
        SetDialogue(messageText, userId)
        return u"請上傳照片，上傳後自動開始辨識\n" + u"辨識無誤，請輸入: #Y\n" + u"辨識有誤，請輸入編號以及正確名稱，\n例如：#Update 0 kelvin"

    elif(len(dialogue) == 3 and dialogue[0] == "#Recognize" and messageText == "#Y"):
        fp = open(os.path.join(fileDir, "wb") +
                  "/" + dialogue[2] + ".jpg", 'rb')
        img = fp.read()
        fp.close()
        images = GetWaittingImages(userId)
        print(images)
        if(len(images) > 0):
            for x in images:
                if(x["name"] != x["predictname"]):
                    lineprocess.UpdateUser(x["name"], x["hash"], x[
                                           "content"], x["representation"])

            users = []
            for x in images:
                if(str(x["name"]) != "Unknown"):
                    users.append(x["name"])

            lineprocess.addRecognizeImage(
                dialogue[1], base64.b64encode(img), users)
        ClearDialogue(userId)
        ofsp.setTrainingData(lineprocess.getTrainingUser(),
                             lineprocess.getTrainingUserData())
        return "辨識已完成"

    elif(len(dialogue) == 3 and dialogue[0] == "#Recognize" and messageText.index("Update") >= 0):
        try:
            ins = messageText.split(" ")
            imageIndex = int(ins[1])
            realName = ins[2]
            UpdateWaittingImages(realName, imageIndex, userId)
            return u"更新完成，如果需要修正，請繼續輸入更新指令\n否則請輸入：#Y"
        except Exception as e:
            print(e)
            ClearDialogue(userId)
            return u"更新失敗，請重新辨識"

    elif (messageText == "#Query"):
        SetDialogue(messageText, userId)
        return "請輸入查詢關鍵字或是人名"

    elif(len(dialogue) == 1 and dialogue[0] == "#Query"):
        SetDialogue(messageText, userId)
        imagesInfo = lineprocess.queryRecognizeImage(messageText)

        if(len(imagesInfo) <= 0):
            ClearDialogue(userId)
            return u"該關鍵字查無資料"

        result = u"查詢結果共 " + str(len(imagesInfo)) + u"筆，如下:\n"
        for x in imagesInfo:
            result += u"(#" + str(x["ImageSeq"]) + u")" + \
                x["ImageDescription"].encode('utf-8') + "\n"

        result += u"如果想查看圖片，請輸入描述前面中的數字\n例如:#1"
        return result
    elif (messageText == "#Help"):
        ClearDialogue(userId)
        return u"梅長蘇:靖王殿下，我想選你~~~~\n(You are my destiny~~~~\n藺晨:ㄟ~~你這死沒良心的\n~~歡迎使用~~\n目前有以下這些指令:\n#CreateNewPerson ----新增人員的功能\n#Recognize ----辨識人員的功能\n#Query ----查詢資料的功能\n目前系統尚未開發完全，\n請小心使用，因為開發中\n請依照指令一步一步使用"
    else:
        ClearDialogue(userId)
        return u""


def GetDialogue(userId):
    global dialogue
    result = None
    for x in dialogue:
        if(x["userId"] == userId):
            result = x
            break

    if(result is None):
        dialogue.append({
            "userId": userId,
            "dialogue": []
        })
        return []

    return result["dialogue"]


def SetDialogue(messageText, userId):
    global dialogue
    for x in dialogue:
        if(x["userId"] == userId):
            x["dialogue"].append(messageText)
            break


def ClearDialogue(userId):
    global dialogue
    global waitting
    for x in dialogue:
        if(x["userId"] == userId):
            x["dialogue"] = []
            break

    for x in waitting:
        if(x["userId"] == userId):
            x["image"] = []
            break


def GetWaittingImages(userId):
    global waitting
    result = None
    for x in waitting:
        if(x["userId"] == userId):
            result = x
            break

    if(result is None):
        return []

    return result["imgaes"]


def SetWaittingImages(imgaes, userId):
    global waitting
    result = None
    for x in waitting:
        if(x["userId"] == userId):
            result = x
            x["imgaes"] = imgaes
            break
    if(result is None):
        waitting.append({
            "userId": userId,
            "imgaes": imgaes
        })


def UpdateWaittingImages(realName, index, userId):
    global waitting
    result = None
    for x in waitting:
        if(x["userId"] == userId):
            result = x
            if(result is not None):
                print(result["imgaes"][index]["name"])
                result["imgaes"][index]["name"] = realName
                print(result["imgaes"][index]["name"])
            break


def ImageProcess(messageContent, messageId, userId):
    floderPath = "wb"
    filename = ""
    dialogue = GetDialogue(userId)
    if(len(dialogue) == 2 and dialogue[0] == "#CreateNewPerson"):
        filename = dialogue[1]
        floderPath = "images/" + dialogue[1]

    # 建立原始空白照片，並從Line回傳Content讀取寫入
    newImageOriginal = Image.new('RGB', (1024, 1024))
    newImageOriginal.save(os.path.join(fileDir, floderPath) + "/" +
                          filename + messageId + ".jpg", "JPEG")
    with open(os.path.join(fileDir, floderPath) + "/" + filename + messageId + ".jpg", 'wb') as fd:
        for chunk in messageContent:
            fd.write(chunk)

    # 將檔案正規化後移除
    newImageOriginal = Image.open(os.path.join(
        fileDir, floderPath) + "/" + filename + messageId + ".jpg")
    nim = newImageOriginal.resize((1024, 1024), Image.BILINEAR)
    newImageOriginal.close()
    os.remove(os.path.join(fileDir, floderPath) +
              "/" + filename + messageId + ".jpg")
    nim.save(os.path.join(fileDir, floderPath) + "/" +
             filename + messageId + ".jpg", "JPEG")


def AnalyzeAnImage(imageName):
    subscription_key = '1673956521414055bf1337e6bc5769f9'
    uri_base = 'westcentralus.api.cognitive.microsoft.com'

    headers = {
        # Request headers.
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.urlencode({
        # Request parameters. All of them are optional.
        'visualFeatures': 'Categories,Description,Color',
        'language': 'en',
    })

    # The URL of a JPEG image to analyze.
    # body = "{'url':'https://9f50b2e1.ngrok.io/wb/6774503903607.jpg'}"

    try:
        print("Execute the REST API call and get the response.")
        # Execute the REST API call and get the response.
        conn = httplib.HTTPSConnection(
            'westcentralus.api.cognitive.microsoft.com')
        conn.request("POST", "/vision/v1.0/analyze?%s" %
                     params, open(os.path.join(fileDir, "wb") + '/' + imageName, 'rb'), headers)
        response = conn.getresponse()
        data = response.read()

        # 'data' contains the JSON data. The following formats the JSON data for display.
        parsed = json.loads(data)
        print("Response:")
        # print(json.dumps(parsed, sort_keys=True, indent=2))
        # print(json["description"]["captions"][0]["text"])
        conn.close()
        return parsed.get("description").get("captions")[0].get("text")

    except Exception as e:
        print('Error:')
        print(e)
    return ""


def putImageText(text, imageName):
    # read_original = cv2.imread(img)
    img = cv2.imread(os.path.join(fileDir, "wb") +
                     "/" + imageName + ".jpg", cv2.IMREAD_COLOR)
    cv2.putText(img, text, (0, 1450),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                color=(0, 0, 0), thickness=5)
    cv2.imwrite(os.path.join(fileDir, "wb") +
                "/" + imageName + "d.jpg", img)


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol():

    def __init__(self):
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def setTrainingData(self, people, imagesInfo):
        self.people = []
        for human in people:
            self.people.append(human)
        self.images = {}
        for imgInfo in imagesInfo:
            self.images[imgInfo["hash"]] = Face(
                np.array(imgInfo["representation"]), imgInfo["identity"])

        print(self.people)
        self.trainSVM()

    def getPredict(self, rep):
        identity = self.svm.predict(rep)[0]
        # GridSearchCV
        decision = self.svm.decision_function(rep)[0]
        identity = lineprocess.verificationPredict(
            identity, decision, len(self.people))
        # decision = abs(self.svm.decision_function(rep)[0][identity])
        # print("----------------------------------")
        # print(identity)
        # print("----------------------------------")
        # print(self.svm.decision_function(rep)[0])
        # print("----------------------------------")
        # print(self.svm.predict_proba(rep)[0])
        # print("----------------------------------")
        # staged = self.svm.staged_predict(rep)
        # for x in staged:
        #    print(x)
        # print("----------------------------------")
        # for x in xrange(0, len(self.people)):
        #    print(self.svm.score(rep, [x]))
        '''
        labels = np.arange(len(self.people))
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            labels, [decision])
        print("----------------------------------")
        print(precision)
        print("----------------------------------")
        print(thresholds)
        '''
        return identity

    def getPredictForAdaboost(self, rep):
        identity = self.svm.predict(rep)[0]
        decision = self.svm.decision_function(rep)[0]
        identity = lineprocess.verificationPredictForAdaboost2(
            identity, decision)
        return identity

    def getPredictForRandom(self, rep):
        identity = self.svm.predict(rep)[0]
        predictProba = self.svm.predict_proba(rep)[0]
        identity = lineprocess.verificationPredictForRandom2(
            identity, predictProba)
        return identity

    def getUserNameFromImage(self, dataURL):
        result = []

        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)
        buf = np.asarray(img)
        rgbFrame = np.zeros((1024, 1024, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]
        annotatedFrame = np.copy(buf)
        bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        if not bbs:
            print("Unable to find a face: {}".format(rgbFrame))

        for bb in bbs:
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

            if alignedFace is None:
                continue

            rep = net.forward(alignedFace)
            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            content = [str(x) for x in alignedFace.flatten()]
            identity = self.getPredict(rep)

            name = ""
            if identity == -1:
                name = "Unknown"
            else:
                name = self.people[identity]
            singleImages = {
                "hash": phash,
                "identity": identity,
                "content": content,
                "representation": rep.tolist(),
                "name": name,
                "predictname": name
            }
            result.append(singleImages)
        return result

    def getData(self):
        X = []
        y = []
        for img in self.images.values():
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
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def trainSVM(self):
        print("+ Training SVM on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            # print(numIdentities)
            if numIdentities <= 1:
                return

            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            print("----------------------------------")
            print(str(datetime.now()))

            self.svm = GridSearchCV(
                SVC(C=1, kernel='rbf', decision_function_shape='ovo'),
                param_grid, cv=2).fit(X, y)

            '''
            self.svm = RandomForestClassifier(
                n_estimators=80, max_features='sqrt').fit(X, y)
            '''
            '''
            self.svm = AdaBoostClassifier(DecisionTreeClassifier(
                max_depth=2), algorithm="SAMME.R", n_estimators=80,
                learning_rate=1.25).fit(X, y)
            '''
            '''
            with open('AdaBoostClassifier_earning_rate=10_n_estimators=5000.pkl', 'wb') as fid:
                cPickle.dump(self.svm, fid)
            '''
            '''
            with open('AdaBoostClassifier_earning_rate=5_n_estimators=500.pkl', 'rb') as fid:
                self.svm = cPickle.load(fid)
            '''
            print(str(datetime.now()))
            print("----------------------------------")

    def processFrame(self, dataURL, identity, messageId):
        self.training = False
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)
        '''
        img.save(os.path.join(fileDir, "temp") + "/test.jpg", "JPEG",
                 quality=80, optimize=True, progressive=True)
        '''
        # buf = np.fliplr(np.asarray(img))
        buf = np.asarray(img)
        rgbFrame = np.zeros((1024, 1024, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        annotatedFrame = np.copy(buf)

        identities = []
        bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        if not bbs:
            print("Unable to find a face: {}".format(rgbFrame))
        # bb = align.getLargestFaceBoundingBox(rgbFrame)
        # bbs = [bb] if bb is not None else []
        annotatedFrame = self.FrameDrawName(buf, bbs, identity, rgbFrame)
        for bb in bbs:
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

            if alignedFace is None:
                continue

            bl = (bb.left(), bb.bottom())
            tr = (bb.right(), bb.top())
            rep = net.forward(alignedFace)
            if len(self.people) == 0:
                identity = -1
            elif self.svm:
                identity = self.getPredict(rep)
                # identity = self.getPredictForAdaboost(rep)
                #identity = self.getPredictForRandom(rep)
            else:
                identity = -1

            if identity == -1:
                cv2.rectangle(annotatedFrame, bl, tr,
                              color=(255, 255, 0), thickness=3)
            else:
                cv2.rectangle(annotatedFrame, bl, tr,
                              color=(102, 204, 255), thickness=3)

            for p in openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP:
                cv2.circle(annotatedFrame, center=landmarks[
                           p], radius=7, color=(102, 204, 255), thickness=-1)

        plt.figure()
        plt.imshow(annotatedFrame)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        img = Image.open(imgdata)

        img.save(os.path.join(fileDir, "wb") + "/" + messageId +
                 "_o.jpg", "JPEG", quality=80, optimize=True, progressive=True)
        img.close()
        image = Image.open(os.path.join(fileDir, "wb") +
                           "/" + messageId + "_o.jpg")

        offical = image.resize((1024, 1024), Image.BILINEAR)
        preview = image.resize((240, 240), Image.BILINEAR)
        image.close()
        os.remove(os.path.join(fileDir, "wb") + "/" + messageId + "_o.jpg")

        offical.save(os.path.join(fileDir, "wb") +
                     "/" + messageId + "_o.jpg", "JPEG")
        preview.save(os.path.join(fileDir, "wb") +
                     "/" + messageId + "_p.jpg", "JPEG")
        plt.close()

    def FrameDrawName(self, buf, bbs, identity, rgbFrame):
        cv2_im = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        draw = ImageDraw.Draw(pil_im, "RGB")
        font = ImageFont.truetype("simhei.ttf", 50)

        i = 0
        for bb in bbs:
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

            if alignedFace is None:
                continue

            rep = net.forward(alignedFace)
            if len(self.people) == 0:
                identity = -1
            elif self.svm:
                identity = self.getPredict(rep)
                # identity = self.getPredictForAdaboost(rep)
                #identity = self.getPredictForRandom(rep)
            else:
                identity = -1

            if identity == -1:
                if len(self.people) == 1:
                    name = self.people[0]
                else:
                    name = "Unknown"
            else:
                name = self.people[identity]

            name = "(" + str(i) + ")" + name
            i = i + 1
            if identity == -1:
                draw.text((bb.left(), bb.top() - 50), name,
                          font=font, fill=(255, 255, 0))
            else:
                draw.text((bb.left(), bb.top() - 50),
                          name, 'rgb(102, 204, 255)', font=font)
                # draw.text((bb.left(), bb.top() - 50),
                #          name, "#2089CC", font=font)
        return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    def TestModelValues(self, testData):
        score = self.svm.score(testData["testX"], testData["testY"])
        print(score)

    def TestUnknownData(self, testData):
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 2
        style = xlwt.XFStyle()
        style.pattern = pattern
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My Sheet')

        row = 0
        for d in testData:
            identity = self.svm.predict(d["representation"])[0]
            decision = self.svm.decision_function(d["representation"])[0]
            pre = lineprocess.getDecisionPredict(
                identity, decision, len(self.people))
            identity = lineprocess.verificationPredictForSVM2(
                identity, decision, len(self.people))

            predictName = self.people[identity]
            if identity == -1:
                predictName = "Unknown"
            lineprocess.updateTestUnknownData(d["dataSeq"], predictName)

            worksheet.write(row, 0, d["username"])
            worksheet.write(row, 1, predictName)
            cell = 2
            for x in xrange(0, len(decision)):
                if x in pre:
                    worksheet.write(row, cell + x, decision[x], style)
                else:
                    worksheet.write(row, cell + x, decision[x])

            row = row + 1

        workbook.save(os.path.join(fileDir, "") + 'SVM_Unknown.xls')

    def TestUnKnownDataForAdaboost(self, testData):
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 2
        style = xlwt.XFStyle()
        style.pattern = pattern
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My Sheet')

        row = 0
        for d in testData:
            iden = self.svm.predict(d["representation"])[0]
            decision = self.svm.decision_function(d["representation"])[0]
            identity = lineprocess.verificationPredictForAdaboost2(
                iden, decision)
            predictName = self.people[identity]
            if identity == -1:
                predictName = "Unknown"
            lineprocess.updateTestUnknownData(d["dataSeq"], predictName)

            worksheet.write(row, 0, d["username"])
            worksheet.write(row, 1, predictName)
            cell = 2
            for x in xrange(0, len(decision)):
                if x == iden:
                    worksheet.write(row, cell + x, decision[x], style)
                else:
                    worksheet.write(row, cell + x, decision[x])

            row = row + 1

        workbook.save(os.path.join(fileDir, "") + 'Adaboost_Unknown.xls')

    def TestUnKnownDataForRandom(self, testData):
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 2
        style = xlwt.XFStyle()
        style.pattern = pattern
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My Sheet')

        row = 0
        for d in testData:
            iden = self.svm.predict(d["representation"])[0]
            predictProba = self.svm.predict_proba(d["representation"])[0]
            identity = lineprocess.verificationPredictForRandom2(
                iden, predictProba)
            predictName = self.people[identity]
            if identity == -1:
                predictName = "Unknown"
            lineprocess.updateTestUnknownData(d["dataSeq"], predictName)

            worksheet.write(row, 0, d["username"])
            worksheet.write(row, 1, predictName)
            worksheet.write(row, 2, predictProba[identity], style)

            cell = 3
            for x in xrange(0, len(predictProba)):
                if x == iden:
                    worksheet.write(row, cell + x, predictProba[x], style)
                else:
                    worksheet.write(row, cell + x, predictProba[x])

            row = row + 1

        workbook.save(os.path.join(fileDir, "") + 'RandomForest_Unknown.xls')

    def TestknownData(self, testData):
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 2
        style = xlwt.XFStyle()
        style.pattern = pattern
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My Sheet')

        row = 0
        for d in testData:
            identity = self.svm.predict(d["representation"])[0]
            decision = self.svm.decision_function(d["representation"])[0]
            pre = lineprocess.getDecisionPredict(
                identity, decision, len(self.people))
            identity = lineprocess.verificationPredictForSVM2(
                identity, decision, len(self.people))

            predictName = self.people[identity]
            if identity == -1:
                predictName = "Unknown"
            lineprocess.updateTestknownData(d["dataSeq"], predictName)

            worksheet.write(row, 0, d["username"])
            worksheet.write(row, 1, predictName)
            cell = 2
            for x in xrange(0, len(decision)):
                if x in pre:
                    worksheet.write(row, cell + x, decision[x], style)
                else:
                    worksheet.write(row, cell + x, decision[x])

            row = row + 1

        workbook.save(os.path.join(fileDir, "") + 'SVM_Known.xls')

    def TestKnownDataForAdaboost(self, testData):
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 2
        style = xlwt.XFStyle()
        style.pattern = pattern
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My Sheet')

        row = 0
        for d in testData:
            iden = self.svm.predict(d["representation"])[0]
            decision = self.svm.decision_function(d["representation"])[0]
            identity = lineprocess.verificationPredictForAdaboost2(
                iden, decision)
            #identity = iden
            predictName = self.people[identity]
            if identity == -1:
                predictName = "Unknown"
            lineprocess.updateTestknownData(d["dataSeq"], predictName)

            worksheet.write(row, 0, d["username"])
            worksheet.write(row, 1, predictName)
            cell = 2
            for x in xrange(0, len(decision)):
                if x == iden:
                    worksheet.write(row, cell + x, decision[x], style)
                else:
                    worksheet.write(row, cell + x, decision[x])

            row = row + 1

        workbook.save(os.path.join(fileDir, "") + 'Adaboost_Known.xls')

    def TestKnownDataForRandom(self, testData):
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 2
        style = xlwt.XFStyle()
        style.pattern = pattern
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My Sheet')

        row = 0
        for d in testData:
            iden = self.svm.predict(d["representation"])[0]
            predictProba = self.svm.predict_proba(d["representation"])[0]
            identity = lineprocess.verificationPredictForRandom2(
                iden, predictProba)
            predictName = self.people[identity]
            if identity == -1:
                predictName = "Unknown"
            lineprocess.updateTestknownData(d["dataSeq"], predictName)

            worksheet.write(row, 0, d["username"])
            worksheet.write(row, 1, predictName)
            worksheet.write(row, 2, predictProba[identity], style)
            cell = 3
            for x in xrange(0, len(predictProba)):
                if x == iden:
                    worksheet.write(row, cell + x, predictProba[x], style)
                else:
                    worksheet.write(row, cell + x, predictProba[x])

            row = row + 1

        workbook.save(os.path.join(fileDir, "") + 'RandomForest_Known.xls')

    def TestMultiData(self, folderName):
        for fileName in os.listdir(fileDir + "/" + folderName):
            fullpath = fileDir + "/" + folderName
            fp = open(os.path.join(fullpath, fileName), "rb")
            img = fp.read()
            fp.close()
            newFileName = fileName.split('.')
            try:
                dataURL = "data:image/jpeg;base64," + base64.b64encode(img)
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
                annotatedFrame = np.copy(buf)
                bbs = align.getAllFaceBoundingBoxes(rgbFrame)
                if not bbs:
                    print("Unable to find a face: {}".format(rgbFrame))
                i = 0
                for bb in bbs:
                    landmarks = align.findLandmarks(rgbFrame, bb)
                    alignedFace = align.align(args.imgDim, rgbFrame, bb, landmarks=landmarks,
                                              landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

                    if alignedFace is None:
                        continue

                    rep = net.forward(alignedFace)
                    if len(self.people) == 0:
                        identity = -1
                    elif self.svm:
                        # identity = self.svm.predict(rep)[0]
                        identity = self.getPredict(rep)
                        #identity = self.getPredictForAdaboost(rep)
                        #identity = self.getPredictForRandom(rep)
                    else:
                        identity = -1
                    bl = (bb.left(), bb.bottom())
                    tr = (bb.right(), bb.top())

                    if identity == -1:
                        cv2.rectangle(annotatedFrame, bl, tr,
                                      color=(255, 255, 0), thickness=5)
                    else:
                        cv2.rectangle(annotatedFrame, bl, tr,
                                      color=(153, 255, 204), thickness=5)

                    for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                        cv2.circle(annotatedFrame, center=landmarks[
                                   p], radius=3, color=(102, 204, 255), thickness=-1)
                    if identity == -1:
                        if len(self.people) == 1:
                            name = self.people[0]
                        else:
                            name = "Unknown"
                    else:
                        name = self.people[identity]

                    name = "(" + str(i) + ")" + name
                    i = i + 1
                    if identity == -1:
                        cv2.putText(annotatedFrame, name, (bb.left(), bb.top(
                        ) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255, 255, 0), thickness=3)
                    else:
                        cv2.putText(annotatedFrame, name, (bb.left(), bb.top(
                        ) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(152, 255, 204), thickness=3)
                plt.figure()
                plt.imshow(annotatedFrame)
                plt.xticks([])
                plt.yticks([])

                imgdata = StringIO.StringIO()
                plt.savefig(imgdata, format='png')
                imgdata.seek(0)
                img = Image.open(imgdata)
                img.save(os.path.join(fileDir, "wb") + "/" + newFileName[
                         0] + "_o.jpg", "JPEG", quality=80, optimize=True, progressive=True)
                img.close()

            except Exception as e:
                print(e)
                continue

    def processFrameForAutoTraining(self, dataURL, identity):
        self.training = True
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
        if not self.training:
            annotatedFrame = np.copy(buf)

        # cv2.imshow('frame', rgbFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        identities = []
        # bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        bbs = [bb] if bb is not None else []
        for bb in bbs:
            # print(len(bbs))
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            if alignedFace is None:
                continue
            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            if phash in self.images:
                identity = self.images[phash].identity
            else:
                rep = net.forward(alignedFace)
                # print(rep)
                self.images[phash] = Face(rep, identity)
                # TODO: Transferring as a string is suboptimal.
                # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                # fx=0.5, fy=0.5).flatten()]
                content = [str(x) for x in alignedFace.flatten()]
                singleImages = {
                    "hash": phash,
                    "identity": identity,
                    "content": content,
                    "representation": rep.tolist()
                }
                return singleImages

    def getAutoTrainingFilePath(self):
        trainingPeople = []
        trainingImages = []
        allFiles = []
        print("--------------getAutoTrainingFilePath---------------")
        for root, dirs, files in os.walk(fileDir + "/images"):
            for d in dirs:
                trainingPeople.append(d)
                self.people.append(d)
                print(d)

            for f in files:
                ide = 0
                for p in trainingPeople:
                    if p in f:
                        ide = trainingPeople.index(p)
                        break

                fp = open(os.path.join(root, f), "rb")
                img = fp.read()
                fp.close()
                # print(f)
                # print("-----------------------------------")
                # print(ide)
                singleImages = self.processFrameForAutoTraining(
                    "data:image/jpeg;base64," + base64.b64encode(img), ide)
                if singleImages is None:
                    print(singleImages)
                    continue
                # print(singleImages)
                trainingImages.append(singleImages)
                self.images[singleImages["hash"]] = Face(
                    np.array(singleImages["representation"]), singleImages["identity"])

        self.trainSVM()
        # print(self.people)
        print("--------------getAutoTrainingFilePath----End--------")

    def incrementalTraining(self):
        trainingPeople = []
        trainingImages = {}
        allFiles = []
        print("--------------incrementalTraining---------------")
        for root, dirs, files in os.walk(fileDir + "/incremental"):
            for d in dirs:
                trainingPeople.append(d)
                self.people.append(d)
                print(d)

            for f in files:
                ide = 0
                for p in self.people:
                    if p in f:
                        ide = self.people.index(p)
                        break

                fp = open(os.path.join(root, f), "rb")
                img = fp.read()
                fp.close()
                print(f)
                print("-----------------------------------")
                print(ide)
                print("-----------------------------------")
                singleImages = self.processFrameForAutoTraining(
                    "data:image/jpeg;base64," + base64.b64encode(img), ide)
                if singleImages is None:
                    print(singleImages)
                    continue

                trainingImages[singleImages["hash"]] = Face(
                    np.array(singleImages["representation"]), singleImages["identity"])
                self.images[singleImages["hash"]] = Face(
                    np.array(singleImages["representation"]), singleImages["identity"])

        X = []
        y = []
        for img in self.images.values():
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
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)

        # print("+ Training SVM on {} labeled images.".format(len(trainingImages)))
        # numIdentities = len(set(y + [-1]))
        # print(numIdentities)
        # if numIdentities <= 1:
        #    return

        self.svm = self.svm.set_params(n_iter=len(self.people)).fit(X, y)
        print("--------------incrementalTraining----End--------")

ofsp = OpenFaceServerProtocol()
awinIns = ""
dialogue = []
waitting = []

if __name__ == "__main__":
    # print "inet_addr: " + inet_addr # ofsp.getAutoTrainingFilePath() #
    # lineprocess.trainingUserFromDir()
    # lineprocess.getKnownUserFromDir()
    # lineprocess.getUnknowUserFromDir()
    ofsp.setTrainingData(lineprocess.getTrainingUser(),
                         lineprocess.getTrainingUserData())
    # lineprocess.ProcessMultiImage("multi")
    # ofsp.TestModelValues(lineprocess.getTestKnownData())
    # ofsp.TestknownData(lineprocess.getTestKnownPredictData())
    # ofsp.TestUnknownData(lineprocess.getTestUnknownData())
    # ofsp.TestMultiData("multi")
    print("----------------------------------")
    # ofsp.TestKnownDataForAdaboost(lineprocess.getTestKnownPredictData())
    print("----------------------------------")
    # ofsp.TestUnKnownDataForAdaboost(lineprocess.getTestUnknownData())
    print("----------------------------------")
    # ofsp.TestKnownDataForRandom(lineprocess.getTestKnownPredictData())
    print("----------------------------------")
    # ofsp.TestUnKnownDataForRandom(lineprocess.getTestUnknownData())
    # lineprocess.test()
    # ofsp.incrementalTraining()

    app.run(
        host=inet_addr,
        port=5000
    )
