import cv2 
import matplotlib.pyplot as plt
#haarcascade_frontalface_default.xml paketinin yüklendiğini ve doğru şekilde okunduğunu kontrol edin.
face_cascade = cv2.CascadeClassifier('C:\\Users\\Zirve\\Downloads\\Video\\haarcascade_frontalface_default.xml')
#Tespit etmek istediğiniz resmin veri yolunu doğru girerek, yüz tespiti yapabilirsiniz.
resim=cv2.imread("C:\\Users\\Zirve\\Desktop\\CNN-Uygulama\\ResimVerilerim\\2.jpeg")

plt.imshow(cv2.cvtColor(resim,cv2.COLOR_BGR2RGB))

gray=cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)

faces = face_cascade.detectMultiScale(gray,1.1,7)

for(x,y,w,h) in faces:
    cv2.rectangle(resim,(x,y),(x+w,y+h),(255,255,0),7)
    
plt.imshow(cv2.cvtColor(resim,cv2.COLOR_BGR2RGB))
