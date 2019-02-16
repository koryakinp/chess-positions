import numpy as np
import os
import glob
import re
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform

piece_symbols = 'prbnkqPRBNKQ'

dictionary = [
  '1', '2', '3', '4', '5', '6', '7', '8', 'p', 'r', 'n',
  'b', 'k', 'q', 'P', 'R', 'N', 'B', 'K', 'Q', '-', 'f']


def fen_to_symbols_one_hot(fen):
    targets = np.array([dictionary.index(symbol) for symbol in fen])
    return np.eye(len(dictionary))[targets]


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


def onehot_from_fen(filename):
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


def fen_from_onehot(one_hot):
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
        y = onehot_from_fen(img)
        x = process_image(img)
        yield x, y


def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)
