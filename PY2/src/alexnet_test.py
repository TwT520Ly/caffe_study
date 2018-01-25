# -*- coding: utf-8 -*-
import caffe
import os
import caffe.io
from caffe import Net, SGDSolver
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

def classification(img, net, transformer):
    im = caffe.io.load_image(img)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    start = time.clock()
    net.forward()
    end = time.clock()
    print 'classification time: %f s' % (end - start)
    labels = np.loadtxt(synset_words, str, delimiter='\t')

    category = net.blobs['prob'].data[0].argmax()

    class_str = labels[int(category)].split(',')
    class_name = class_str[0]
    # text_font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
    cv2.putText(im, class_name, (0, im.shape[0]), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 显示结果
    plt.imshow(im, 'brg')
    plt.show()

# CPU or GPU
caffe.set_mode_cpu()
# caffe.set_mode_gpu()
caffe_root = '../'
caffemodel = caffe_root + ''
deploy = caffe_root + ''
net = caffe.Net(deploy, caffemodel, caffe.TEST)

mu = np.load(caffe_root+'data/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

while(1):
    img_num = raw_input("Enter images number:")
    if img_num == '': break
    img = img_root + '{:0>6}'.format(img_num) + '.jpg'
    classification(img, net, transformer, synset_words)