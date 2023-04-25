import cv2 #importaçoes 
import mediapipe as mp
import numpy as np
from keras.models import load_model
from gtts import gTTS
from playsound import playsound
import time 

t = 2
time.sleep(t) # exemplo 


cap = cv2.VideoCapture(0) #abrir a camera usnado o open cv

hands = mp.solutions.hands.Hands(max_num_hands=1) # método do mediapipe (capitura um mão)

classes = ['A','B','C','D','E'] # esse array contem as classes do arquivo keras_model as label
model = load_model('keras_model.h5') # importando o modelo que foi feito no site do teachble machine
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # dados de entrada do modelo 

while True: #loop finito
    success, img = cap.read() # aqui e feita a leitura da imagem da webcam
    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    results = hands.process(frameRGB) # processa freme por freme 
    handsPoints = results.multi_hand_landmarks # capta a mão por completo 
    h, w, _ = img.shape 

    if handsPoints != None: # essa condicional verifica se existe uma mão na imgem 
        for hand in handsPoints: # essa sequência de if dimensiona o Balding box da mão
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(img, (80,10), (200,100), (225,0,0),-1) # já dimensionado agora o desenho usando o cv2
            #x_min-50, y_min-50
            #x_max+50, y_max+1
            #0, 255, 0
            try:
                imgCrop = img[y_min-50:y_max+50,x_min-50:x_max+50] # a imagem recortada com a mão
                imgCrop = cv2.resize(imgCrop,(224,224)) # aqui redimenciona a imagem e transforma ela em uma array
                imgArray = np.asarray(imgCrop) #faz a normalização 
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data) # agora com a imagem tratada ele faz a predição de qual classe a forma da mão pertence
                indexVal = np.argmax(prediction) # por meio da probabilidade ele indica o indice do array (classes) na linhaclea 10
                indice = classes[indexVal]
                #print(classes[indexVal])
                cv2.putText(img,classes[indexVal],(100,100),cv2.FONT_HERSHEY_COMPLEX,4,(225,225,255),5) #puttext cria um texto dentro da imagem já estranindo da variavel classes qual e a label predominate
                #x_min-120,y_min-1
                #3,(0,0,255),5)
                print("letra gerada " + indice)
                
                # if indice == 'A':
                #     audio_text = ('A')
                #     tts = gTTS(audio_text,lang='pt-br')
                #     tts.save("audioA.mp3")
                #     playsound("audioA.mp3")

                while indice:
                    if indice == 'A':
                        audio_text = ('A')
                        tts = gTTS(audio_text,lang='pt-br')
                        playsound("../public/audios/audioA.mp3")
                        break
                    if indice == 'B':
                        audio_text = ('B')
                        tts = gTTS(audio_text,lang='pt-br')
                        playsound("../public/audios/audioB.mp3")
                        break
                    if indice == 'C':
                        audio_text = ('C')
                        tts = gTTS(audio_text,lang='pt-br')
                        playsound("../public/audios/audioC.mp3")
                        break
                    if indice == 'D':
                        audio_text = ('D')
                        tts = gTTS(audio_text,lang='pt-br')
                        playsound("../public/audios/audioD.mp3")
                        break
                    if indice == 'E':
                        audio_text = ('E')
                        tts = gTTS(audio_text,lang='pt-br')
                        playsound("../public/audios/audioE.mp3")
                        break    
            except:
                continue

    cv2.imshow('Imagem',img)
    if cv2.waitKey(5) & 0xFF == 27:
       break
cap.release()