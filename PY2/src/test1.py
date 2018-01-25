# -*- coding: utf-8 -*-
import caffe
import os
import caffe.io
from caffe import Net, SGDSolver

# print os.path.exists('../train/lenet_solver.prototxt')

solver = caffe.SGDSolver('../train/lenet_solver.prototxt')
solver.net.copy_from('../model/_iter_10000.caffemodel')
solver.solve()
train_net = solver.net
test_net = solver.test_nets[0]




