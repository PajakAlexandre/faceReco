#!/usr/bin/env python
import cv2
import pickle
import numpy as np
import common as c

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
fullbody_cascade= cv2.CascadeClassifier('haarcascade_fullbody.xml')
upperbody_cascade= cv2.CascadeClassifier('haarcascade_upperbody.xml')
lowerbody_cascade= cv2.CascadeClassifier('haarcascade_lowerbody.xml')

#fonction de reconnaissance
recognizer=cv2.face.LBPHFaceRecognizer_create()
#fichier de données d'appren tissage
recognizer.read("trainner.yml")
id_image=0
color_info=(255, 255, 255)
color_ko=(0, 0, 255)
color_ok=(0, 255, 0)

#récupération de noms qui vont avec les id
with open("labels.pickle", "rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k, v in og_labels.items()}

cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(c.min_size, c.min_size))
    fullbodys=fullbody_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(c.min_size, c.min_size))
    upperbodys=upperbody_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(c.min_size, c.min_size))
    lowerbodys=lowerbody_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(c.min_size, c.min_size))

    for (x, y, w, h) in faces:
        #récupère uniqument le visage
        roi_gray=cv2.resize(gray[y:y+h, x:x+w], (c.min_size, c.min_size))
        #indice de confiance de correspondance
        id_, conf=recognizer.predict(roi_gray)
        if conf<=115: #si confiance ok on récupère le nom de la personne
             color=color_ok
             name=labels[id_]
        else:
            color=color_ko
            name="Inconnu"
        label=name+" "+'{:5.2f}'.format(conf)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    for (x, y, w, h) in fullbodys:
        color=color_ko
        label="fullBody"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    for (x, y, w, h) in upperbodys:
        color=color_ko
        label="upperBody"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    for (x, y, w, h) in lowerbodys:
        color=color_ko
        label="lowerBody"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_info, 2)
    cv2.imshow('Drone-sécurité', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100):
            ret, frame=cap.read()

cv2.destroyAllWindows()
print("Fin")