import numpy as np
import os
import glob
import re
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from keras.models import Sequential
from keras import Input, layers
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import add
from utils import *

train_size = 10000
test_size = 3000

train = glob.glob("./data/train/*.jpeg")
test = glob.glob("./data/test/*.jpeg")

longest_fen = len(fen_from_filename(max(train + test, key=len)))
dictionery_size = len(dictionary)

shuffle(train)
shuffle(test)

train = train[:train_size]
test = test[:test_size]

inputs1 = Input(batch_shape=(1, 832))
fe = Dense(64, activation='relu')(inputs1)
inputs2 = Input(batch_shape=(1, longest_fen, dictionery_size))
se = LSTM(64)(inputs2)
decoder1 = add([fe, se])
output = Dense(64, activation='softmax')(decoder1)
outputs = Dense(dictionery_size, activation='softmax')(output)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for idx, image in enumerate(train):
    fen = fen_from_filename(image)
    img_features = onehot_from_fen(fen).reshape(1, 832)
    fen = fen + 'f'
    seen = ''

    loss = np.array([])

    for i, symbol in enumerate(fen):

        fen_one_hot = np.zeros((longest_fen, dictionery_size))
        y = fen_to_symbols_one_hot(symbol)

        if(seen != ''):
            fen_one_hot[:i] = fen_to_symbols_one_hot(seen)

        fen_one_hot = np.expand_dims(fen_one_hot, axis=0)
        res = model.train_on_batch([img_features, fen_one_hot], y)
        loss = np.append(loss, res[0])

        if(symbol == 'f'):
            break

        seen += symbol

    print("\r {}/{} | {:1.5f}".format(idx, train_size, loss.mean()), end='')
