import cv2 
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
emotion_dict = {0: "Kizgin", 1: "İgrenmis", 2: "Korkmus", 3: "Mutlu", 4: "Dogal", 5: "Uzgun", 6: "Saskin"}
json_file = open('Model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")
face_cascade = cv2.CascadeClassifier('C:\\Users\\Zirve\\Downloads\\Video\\haarcascade_frontalface_default.xml')
resim=cv2.imread("C:\\Users\\Zirve\\Desktop\\CNN-Uygulama\\ResimVerilerim\\1.jpeg")
plt.imshow(cv2.cvtColor(resim,cv2.COLOR_BGR2RGB))

gray=cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)

faces = face_cascade.detectMultiScale(gray,1.1,7)

for(x,y,w,h) in faces:
    cv2.rectangle(resim,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray_frame = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
    
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(gray, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
plt.imshow(cv2.cvtColor(resim,cv2.COLOR_BGR2RGB))
