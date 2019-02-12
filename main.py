import numpy as np
import os
import glob
import re
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D


train_img_list = glob.glob("./data/train/*.jpeg")
test_img_list = glob.glob("./data/test/*.jpeg")

shuffle(train_img_list)
shuffle(test_img_list)

train_size = 500
test_size = 500

piece_symbols = 'prbnkqPRBNKQ'


def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size/8)
    img_read = io.imread(img, channels=3)
    img_read = transform.resize(
      img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)


def fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


def build_labels(filename):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = fen_from_filename(filename)
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if(char in '12345678'):
            output = np.append(
              output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)

    return output


def build_fen_from_one_hot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output


def import_data():
    train_img = np.empty((0, 25, 25, 3))
    test_img = np.empty((0, 25, 25, 3))
    train_label = np.empty((0, 13))
    test_label = np.empty((0, 13))

    train_count = len(train_img_list)
    test_count = len(test_img_list)

    for i, img in enumerate(train_img_list[:train_size]):

        print("\rProcessing Training Set: {}/{}".format(
          i + 1, train_count), end="")
        train_label = np.append(train_label, build_labels(img), axis=0)
        train_img = np.append(train_img, process_image(img), axis=0)

    print("\r")

    for i, img in enumerate(test_img_list[:test_size]):
        print("\rProcessing Test Set: {}/{}".format(
          i + 1, test_count), end="")
        test_label = np.append(test_label, build_labels(img), axis=0)
        test_img = np.append(test_img, process_image(img), axis=0)

    print("\r")

    train_fens = [fen_from_filename(f) for f in train_img_list[:train_size]]
    test_fens = [fen_from_filename(f) for f in test_img_list[:test_size]]

    return np.array(train_img), np.array(
      test_img), np.array(train_label), np.array(
      test_label), np.array(train_fens), np.array(test_fens)

xtrain, xtest, ytrain, ytest, train_fens, test_fens = import_data()

print(xtrain.shape)
print(ytrain.shape)

X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(
  xtrain, ytrain, test_size=0.2, random_state=1)

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(25, 25, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(13))
model.add(Activation('softmax'))

model.compile(
  loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
  X_Train, Y_Train, batch_size=64,
  epochs=3, verbose=1, shuffle=True,
  validation_data=(X_Validation, Y_Validation))

res = model.predict(xtest)
res = res.argmax(axis=1)
res = res.reshape(-1, 8, 8)

pred_fens = np.array([build_fen_from_one_hot(one_hot) for one_hot in res])
accuracy = (pred_fens == test_fens).astype(float).mean()

print(accuracy)
