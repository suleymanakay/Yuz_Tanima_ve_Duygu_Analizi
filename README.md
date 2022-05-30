# C-NN_ile_Yuz_Tespiti_ve_Duygu_Analizi_Tespiti 

### Yuklenecek paketler:
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### Kaggleden yuklenmesi gereken veri seti fer2013
- yÃ¼klenilen verilerin  projeye dahil edilmesi
- https://www.kaggle.com/msambare/fer2013

###  Duygu Egitim Dedektörünü baslatın (Train Emotion detector)
- icindeki tümm yuz ifadesi goruntuleri ile FER2013 Verisetine aiitir.
- command --> python TranEmotionDetector.py

islemcinize bagli olarak birkac saat surecektir. (8 GB RAM'li i5 7.nesil islemcide yaklasik 4 saatimi aldi)
Egitimden sonra, eÄitilmiÅ model yapÄ±sÄ±nÄ± ve aÄÄ±rlÄ±klarÄ±n proje dizininizde saklandÄ±ÄÄ±nÄ± gÃ¶receksiniz.
emotion_model.json
emotion_model.h5
yukaridaki 2 değeri model dosyası altında kaydedin

### Duygu dedektorunu calisitirin (emotion detection) 
python TestEmotionDetector.py
