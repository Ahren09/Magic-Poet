#coding:utf8
import numpy as np
import os


def get_data(opt):
    if os.path.exists(opt.pickle_path):
        data_load = np.load(opt.pickle_path, allow_pickle=True)
        data = data_load['data']
        ix2word, word2ix = data_load['ix2word'].item(), data_load['word2ix'].item()
        return data, word2ix, ix2word


'''
# Load and display one example

PATH = './tang.npz'
data_loader = np.load(PATH, allow_pickle=True)
data = data_loader['data']
ix2word = data_loader['ix2word'].item()

poem = data[0]
poem_txt = [ix2word[i] for i in poem]
print(''.join(poem_txt))
'''

