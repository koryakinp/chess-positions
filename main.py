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

train, valid = train_test_split(train_img_list, test_size=0.2, random_state=1)
test = test_img_list

test_size = 3000

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


def get_fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


def build_labels(filename):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = get_fen_from_filename(filename)
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


def train_gen(features, labels, batch_size):
    for i, img in enumerate(features):
        y = build_labels(img)
        x = process_image(img)
        yield x, y


def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)


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

model.fit_generator(train_gen(train, None, 64), steps_per_epoch=10000)

res = (
  model.predict_generator(pred_gen(test[:test_size], 64), steps=test_size)
  .argmax(axis=1)
  .reshape(-1, 8, 8)
)

pred_fens = np.array([build_fen_from_one_hot(one_hot) for one_hot in res])
test_fens = np.array([get_fen_from_filename(fn) for fn in test[:test_size]])

final_accuracy = (pred_fens == test_fens).astype(float).mean()

print("Final Accuracy: {:1.5f}%".format(final_accuracy))
