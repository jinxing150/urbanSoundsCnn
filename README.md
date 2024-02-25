# Spectrogram Classification with UrbanSound8K Dataset

In this project we used the dataset of UrbanSound8K which you can access here:  
https://urbansounddataset.weebly.com/urbansound8k.html

This dataset contains 8732 labeled sound excerptsvof urban sounds from 10 classes: air conditioner, car horn, children playing, dog bark, drilling, enginge idling, gun shot, jackhammer, siren, and street music. The classes are drawn from the urban sound taxonomy. All excerpts are taken from field recordings uploaded to www.freesound.org.
8732 audio files of urban sounds (see description above) in WAV format. The sampling rate, bit depth, and number of channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).
The UrbanSound8k dataset used for model training, can be downloaded from the following link: 
https://urbansounddataset.weebly.com/

**Quick Notice: This model has been created by using 2 different notebooks, one directly from the spectograms and the other one from numerical values obatined by spectrograms. According to the data, first model has produced 10% more successful results (epoch = 20, batchsize = 250, if the value is assigned) than the second model.**

## **Preprocessing**

**Used Libraries:** 

```,
import pandas as pd
import numpy as np
import librosa
import cv2
!pip install opensoundscape
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
import matplotlib.pyplot as pltimport os
```


**Creating Spectrograms:**
```
def createSpectogram2(): 
    # This method creates stpectogram from audio file, converts spectrogram to image and saves spectrogram image to folder by class
    for x in range(len(data)):
        new_path = path+"fold"+data.iloc[x,5].astype(str)+"/"+data.iloc[x,0]
        file_name = data.iloc[x,0]
        file_name = file_name[:-4].strip().replace(" ","")
        class_ID = data.iloc[x,6]
        audio = Audio.from_file(new_path)
        spectrogram = Spectrogram.from_audio(audio)
        spectogram.append(spectrogram)
        image = spectrogram.to_image(shape=image_shape,invert=True)
        
        # image_path = "folders/fold"+class_ID.astype(str)
        # image.save(f"{image_path}/{file_name}.png")
        if(x==(len(data)-1)):
            plt.imshow(image)
    
     
createSpectogram2()
```

## Building Up A Convolutional Neural Network Model

**Used Libraries:**

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from PIL import Image
from keras.utils.np_utils import to_categorical 
import librosa
import cv2
import os
!pip install librosa
!pip install opencv-python
!pip install tensorflow
!pip install keras
```


**Resizing 2.1:**
Resizing the X Values:

```
path = "UrbanSound8K/audio/"
dim = (64,64)
for x in range(0,len(data1)):
    new_path = path+"fold"+data1.iloc[x,5].astype(str)+"/"+data1.iloc[x,0]
    file_name = data1.iloc[x,0]
    file_name = file_name[:-4].strip().replace(" ","")
     
    y,sr = librosa.load(new_path)
    spec = librosa.feature.melspectrogram(y=y,sr=sr)
    specDB= librosa.amplitude_to_db(spec,ref= np.max)
    resized_specDB = cv2.resize(specDB,dim, interpolation = cv2.INTER_AREA)
    spectrogram.append(resized_specDB)
                        
spectrogram = np.array(spectrogram)
```

**Resizing 2.2:**
Resizing the Y Values:


```
y2 = []
foldNumber = 0
for folder in os.listdir('folders'):
    for image in os.listdir('folders/'+folder):
        img = Image.open('folders/'+folder+'/'+image)
        img = img.resize((64,64))
        img_s = img.getdata()
        img_m = np.array(img_s)
        img_m = img_m/255
        data.append(img_m)
        y2.append(foldNumber) 
    foldNumber=foldNumber+1  

y1 = np.array(y1)    
data = np.array(data).reshape(-1,64,64,1)
y2 = np.array(y2) 
from sklearn.model_selection import train_test_split
```


** Training the Model (1/2) 2.1: **
Training:
```
X_train, X_test, y_train, y_test = train_test_split(spectrogram, y1, test_size=0.33, random_state=42)
y_one_hot =  to_categorical(y_train, num_classes = 10)
y_t_one_hot = to_categorical(y_test, num_classes = 10)
model = Sequential()
    
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])   
```


** Training the Model (1/2) 2.2: **
Accuracy:
```
history = model.fit(X_train,y_one_hot, batch_size=250, epochs=20,validation_data=(X_test, y_t_one_hot))
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

Result:


![image](https://user-images.githubusercontent.com/76561240/194776647-11a50bcb-150b-4557-ba6a-56af2dbf1205.png)


** Training the Model (2/2) 2.1:**
Training:


```
X_train2, X_test2, y_train2, y_test2 = train_test_split(data, y2, test_size=0.33, random_state=42)
y_one_hot2 =  to_categorical(y_train2, num_classes = 10)
y_t_one_hot2 = to_categorical(y_test2, num_classes = 10)
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64,64,1)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))
model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) 
```


** Training the Model (2/2) 2.2: **
Accuracy:


```
history2 = model2.fit(X_train2,y_one_hot2, batch_size=250, epochs=20,validation_data=(X_test2, y_t_one_hot2))

plt.plot(history2.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```


Result:


![image](https://user-images.githubusercontent.com/76561240/194776775-46e60228-5056-4ba3-803a-24dbc2b882c0.png)
