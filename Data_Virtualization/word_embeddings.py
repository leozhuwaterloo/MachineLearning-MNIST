import os
import urllib
import requests
import tensorflow as tf
import zipfile
import collections
import random

URL = 'http://mattmahoney.net/dc/'

PATH = os.getcwd()
LOG_DIR = os.path.join(PATH, 'T-Sne_Log')
DATA_DIR = os.path.join(PATH, 'Data')

VOCABULARY_SIZE = 50000
EMBEDDING_SIZE = 10


def get_data(file_name, expected_bytes):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        if not tf.gfile.Exists(DATA_DIR):
            tf.gfile.MakeDirs(DATA_DIR)
        file_path, _ = urllib.request.urlretrieve(URL + file_name, file_path)
    stat_info = os.stat(file_path)
    if stat_info.st_size == expected_bytes:
        print('Found and verified', file_name)
    else:
        print(stat_info.st_size)
        raise Exception(
            'Failed to verify ' + file_name + '. Can you get to it with a browser?')
    return file_path


def read_data(data_file_zip):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(data_file_zip) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
