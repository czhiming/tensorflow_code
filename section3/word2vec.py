#-*- coding:utf8 -*-
'''
Created on Jul 11, 2017

@author: czm
'''
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = "http://mattmahoney.net/dc/"

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    return filename

filename = maybe_download('text8.zip', 31344016)




if __name__ == '__main__':
    pass