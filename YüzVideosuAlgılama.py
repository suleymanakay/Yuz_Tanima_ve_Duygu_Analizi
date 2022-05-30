import cv2

cap = cv2.VideoCapture('C:\\Users\\Zirve\\Downloads\\Video\\kk.mp4')
face_cascade = cv2.CascadeClassifier('C:\\Users\\Zirve\\Downloads\\Video\\haarcascade_frontalface_default.xml')
while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame,1.1, minNeighbors = 7)
            
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(0,0,255),10)
        cv2.imshow("Yuz Algila", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()