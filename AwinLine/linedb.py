#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
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
import MySQLdb
from array import array

db = MySQLdb.connect(host="localhost", user="root",
                     passwd="1234", db="face", charset="utf8")


def CheckDbConnect():
    try:
        db.ping()
    except Exception as e:
        print(e)

    return MySQLdb.connect(host="localhost", user="root", passwd="1234", db="face", charset="utf8")


def addTrainingUser(identity, username):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO usermain VALUES (%d,'%s');" % (identity, username))
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()


def addTrainingUserPhoto(identity, hashValue, content, representation):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO userphoto (`identity`,`hash`,`content`,`representation`)"
                       "VALUES (%d,'%s','%s','%s');" % (identity, hashValue, content, representation))
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()


def addKnownUser(identity, username, representation):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO test_known (`representation`,`user_name`,`identity`)"
                       "VALUES ('%s','%s',%d);" % (representation, username, identity))
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()


def addUnknownUser(username, representation):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO test_unknown (`representation`,`user_name`)"
                       "VALUES ('%s','%s');" % (representation, username))
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()


def getUserIdentity(username):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT "
                       "identity "
                       "FROM "
                       "usermain "
                       "WHERE "
                       "username = '%s' " % (username))
        identity = 0
        for i in range(0, 1):
            record = cursor.fetchone()
            identity = record[0]
            if identity is None:
                identity = 0
            else:
                identity = int(identity)
        return identity
    except Exception as e:
        print(e)
    cursor.close()


def judmentUserIsExsist(username):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT "
                       "COUNT( * ) "
                       "FROM "
                       "usermain "
                       "WHERE "
                       "username = '%s' " % (username))
        identity = 0
        for i in range(0, 1):
            record = cursor.fetchone()
            identity = record[0]
            if identity is None:
                identity = 0
            else:
                identity = int(identity)
        return identity > 0
    except Exception as e:
        print(e)
    cursor.close()


def getUserMaxIdentity():
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT MAX(identity) FROM usermain")
        identity = 0
        for i in range(0, 1):
            record = cursor.fetchone()
            identity = record[0]
            if identity is None:
                identity = 0
            else:
                identity = int(identity) + 1
        return identity
    except Exception as e:
        print(e)
    cursor.close()


def getTrainingUser():
    db = CheckDbConnect()
    cursor = db.cursor()
    result = []
    try:
        cursor.execute("SELECT "
                       "identity, "
                       "username  "
                       "FROM "
                       "usermain "
                       "ORDER BY "
                       "identity")
        rc = cursor.rowcount
        for i in range(0, rc):
            record = cursor.fetchone()
            username = record[1].encode()
            username = username.decode('UTF-8')
            result.append(username)

    except Exception as e:
        print(e)
    cursor.close()
    return result


def getTrainingUserData():
    db = CheckDbConnect()
    cursor = db.cursor()
    result = []
    try:
        cursor.execute("SELECT "
                       "m.identity, "
                       "m.username, "
                       "p.`hash`, "
                       "p.content, "
                       "p.representation "
                       "FROM "
                       "usermain m "
                       "INNER JOIN userphoto p ON m.identity = p.identity")
        # 取得資料總筆數
        rc = cursor.rowcount

        # 一次取出一筆資料
        for i in range(0, rc):
            record = cursor.fetchone()
            identity = int(record[0])
            username = record[1]
            username = username.decode('UTF-8')
            hashValue = record[2].encode()
            content = record[3].encode().split(',')
            representation = record[4].encode().split(',')
            representation = map(float, representation)
            result.append({
                "hash": hashValue,
                "identity": identity,
                "content": content,
                "representation": representation
            })

    except Exception as e:
        print(e)
    cursor.close()
    return result


def addRecognizeImage(imageDescription, imageBase64, users):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        userSql = ""
        for i in range(0, len(users)):
            if i == 0:
                userSql += (" SELECT '%s' AS username " % (users[i]))
            else:
                userSql += (" UNION ALL SELECT '%s' AS username " % (users[i]))

        cursor.execute("INSERT INTO recognize_images (`image_description`,`image_base64`) "
                       "VALUES ('%s','%s'); " % (imageDescription, imageBase64))

        cursor.execute("SELECT LAST_INSERT_ID( );")

        for i in range(0, 1):
            record = cursor.fetchone()
            imageseq = int(record[0])
            sql = ("INSERT INTO r_user_recognize (`imageseq`,`identity`) SELECT "
                   "%d, "
                   "m.identity "
                   "FROM "
                   "usermain m "
                   "INNER JOIN ( " % (imageseq))

            sql = sql + userSql + " ) a ON m.username = a.username "

            cursor.execute(sql)
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()


def queryRecognizeImage(keyWord):
    db = CheckDbConnect()
    cursor = db.cursor()
    result = []
    try:
        sqlCommand = ("SELECT "
                      "* "
                      "FROM "
                      "( "
                      "SELECT DISTINCT "
                      "i.imageseq, "
                      "i.image_description, "
                      "i.create_time "
                      "FROM "
                      "recognize_images i "
                      "INNER JOIN r_user_recognize r ON i.imageseq = r.imageseq "
                      "INNER JOIN usermain m ON r.identity = m.identity "
                      "WHERE "
                      "i.image_description LIKE %s "
                      "OR m.username = '%s' "
                      ") A "
                      "ORDER BY "
                      "A.create_time " % ("'%" + keyWord + "%'", keyWord))
        cursor.execute(sqlCommand)
        # 取得資料總筆數
        rc = cursor.rowcount

        # 一次取出一筆資料
        for i in range(0, rc):
            record = cursor.fetchone()
            imageseq = int(record[0])
            image_description = record[1]
            result.append({
                "ImageSeq": imageseq,
                "ImageDescription": image_description
            })

    except Exception as e:
        print(e)
    cursor.close()
    return result


def getQueryImage(imageSeq):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT "
                       "image_base64 "
                       "FROM "
                       "recognize_images "
                       "WHERE "
                       "imageseq = %d " % (int(imageSeq)))
        for i in range(0, 1):
            record = cursor.fetchone()
            return record[0].encode()
    except Exception as e:
        print(e)
    cursor.close()


def getTestKnownData():
    db = CheckDbConnect()
    cursor = db.cursor()
    result = {}
    testX = []
    testY = []

    try:
        cursor.execute("SELECT "
                       "representation, "
                       "identity "
                       "FROM "
                       "test_known")
        # 取得資料總筆數
        rc = cursor.rowcount
        # 一次取出一筆資料
        for i in range(0, rc):
            record = cursor.fetchone()
            identity = int(record[1])
            representation = record[0].encode().split(',')
            representation = map(float, representation)
            testX.append(representation)
            testY.append(identity)

    except Exception as e:
        print(e)
    cursor.close()
    result = {
        "testX": testX,
        "testY": testY
    }
    return result


def getTestKnownPredictData():
    db = CheckDbConnect()
    cursor = db.cursor()
    result = []
    try:
        cursor.execute("SELECT "
                       "data_seq, "
                       "representation, "
                       "user_name "
                       "FROM "
                       "test_known "
                       "ORDER BY "
                       "identity ")
        #"LIMIT 100 OFFSET 0")
        # 取得資料總筆數
        rc = cursor.rowcount
        # 一次取出一筆資料
        for i in range(0, rc):
            record = cursor.fetchone()
            dataSeq = int(record[0])
            representation = record[1].encode().split(',')
            representation = map(float, representation)
            username = record[2]
            result.append({
                "dataSeq": dataSeq,
                "representation": representation,
                "username": username
            })

    except Exception as e:
        print(e)
    cursor.close()

    return result


def getTestUnknownData():
    db = CheckDbConnect()
    cursor = db.cursor()
    result = []
    try:
        cursor.execute("SELECT "
                       "data_seq, "
                       "representation, "
                       "user_name, "
                       "predict_name "
                       "FROM "
                       "test_unknown "
                       "ORDER BY "
                       "user_name ")
        #"LIMIT 100 OFFSET 0")
        # 取得資料總筆數
        rc = cursor.rowcount
        # 一次取出一筆資料
        for i in range(0, rc):
            record = cursor.fetchone()
            dataSeq = int(record[0])
            representation = record[1].encode().split(',')
            representation = map(float, representation)
            username = record[2]
            result.append({
                "dataSeq": dataSeq,
                "representation": representation,
                "username": username
            })

    except Exception as e:
        print(e)
    cursor.close()

    return result


def updateTestUnknownData(dataSeq, predictName):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE test_unknown "
                       "SET predict_name = '%s' "
                       "WHERE "
                       "data_seq = %d" % (predictName, dataSeq))
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()


def updateTestknownData(dataSeq, predictName):
    db = CheckDbConnect()
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE test_known "
                       "SET predict_name = '%s' "
                       "WHERE "
                       "data_seq = %d" % (predictName, dataSeq))
        db.commit()
    except Exception as e:
        print(e)
    cursor.close()
