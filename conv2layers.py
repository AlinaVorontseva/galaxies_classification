import astropy.io.fits as pyfits

import os
import pickle
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense, AveragePooling2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics

def Make_history_plots(history,name):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mean_squared_error')
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'])

    plt.subplot(1, 3, 3)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'])
    
    plt.savefig('history/history_'+name, format='png')
    
parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='model')
args = parser.parse_args()

label_file = "R_filtrid/BD_decomp_asi.txt"
galaxies_info = pd.read_table(label_file,header=None,index_col=False,skiprows=1)
part1 = galaxies_info.loc[galaxies_info[1].isnull(),[0,2,3,4,5,6]]
part1.columns = ["cnr","kval","varb","dust","popul","rescale"]
part2 = galaxies_info.loc[galaxies_info[1].notnull(),[0,1,2,3,4,5]]
part2.columns = ["cnr","kval","varb","dust","popul","rescale"]
galaxies_info = part1.append(part2)

labels = np.array(galaxies_info['kval'])
labels = np.repeat(labels,4)

cropped_galaxies = pickle.load( open( "cropped_galaxies_v2.p", "rb" ) )
cropped_galaxies = sum(cropped_galaxies, [])

galaxies = []
for arr in cropped_galaxies:    
    galaxies.append(arr[:, :, np.newaxis])
galaxies_array = np.array(galaxies)

X_train, X_test, y_train, y_test = train_test_split(galaxies_array, labels, test_size=0.2, random_state=2)

im_size = 100
batch_size = 32

x = Input(shape=(im_size, im_size, 1))
h = Conv2D(96, (8, 8))(x)
#h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Conv2D(96, (8, 8))(h)
#h = BatchNormalization()(h)
h = Activation('relu')(h)
h = MaxPooling2D(pool_size = (3, 3))(h)
#h = Dropout(rate=0.25)(h)
h = Flatten()(h)
h = Dense(24)(h)
#h = BatchNormalization()(h)
h = Activation('relu')(h)
#h = Dropout(rate=0.5)(h)
p = Dense(1)(h)

model = Model(inputs=x, outputs=p)
model.compile(loss='mse', 
              optimizer=Adam(lr=0.00001), 
              metrics=[metrics.mean_squared_error, metrics.mean_absolute_error])
model.summary()

history = model.fit(X_train, y_train,
          epochs=10,
          batch_size=batch_size,
          validation_split=0.2)

Make_history_plots(history,args.name)
model.save('saved_models/model_' + args.name + '.h5')