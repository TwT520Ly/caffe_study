# -*- coding: utf-8 -*-
import caffe
import os
import caffe.io
from caffe import Net, SGDSolver



solver_proto ='../mnist/solver.prototxt'  # 参数文件

print os.path.exists(solver_proto)
# solver = caffe.SGDSolver('../train/lenet_solver.prototxt')
# solver = caffe.SGDSolver('C:/Users/miconron/Desktop/mnist/solver.prototxt')
solver = caffe.SGDSolver(solver_proto)
# solver.net.copy_from('../model/_iter_10000.caffemodel')
solver.solve()
train_net = solver.net
test_net = solver.test_nets[0]
