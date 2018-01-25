# -*- coding: utf-8 -*-
import caffe
import os
import caffe.io
from caffe import Net, SGDSolver

print os.path.exists('../train/lenet_solver.prototxt')

solver = caffe.SGDSolver('../train/lenet_solver.prototxt')
solver.solve()




