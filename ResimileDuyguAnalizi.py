import cv2 
import matplotlib.pyplot as plt
from deepface import DeepFace
face_cascade = cv2.CascadeClassifier('C:\\Users\\Zirve\\Downloads\\Video\\haarcascade_frontalface_default.xml')

resim=cv2.imread("C:\\Users\\Zirve\\Desktop\\CNN-Uygulama\\ResimVerilerim\\3.jpeg")
gray=cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
#print (faceCascade empty() )
faces = face_cascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(resim,(x,y),(x+w,y+h),(0,255,0),7)
    
    facedet=DeepFace.analyze(resim)
    font=cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(resim,
            facedet['dominant_emotion'],
            (20,100),
            font,2,
            (0,0,255),
            2,
            cv2.LINE_4);
plt.imshow(cv2.cvtColor(resim,cv2.COLOR_BGR2RGB))

   
