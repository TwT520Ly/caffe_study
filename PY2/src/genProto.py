# encoding: utf-8
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import proto
from caffe import to_proto

root = '../mnist/' #根目录
root1 = '../train/'
train_list = root + 'train/train.txt' #训练图片列表
test_list = root + 'test/test.txt' #测试图片列表
train_proto = root + 'train.prototxt'#训练配置文件
test_proto = root + 'test.prototxt'#测试配置文件
solver_proto = root + 'solver.prototxt'#参数文件

#编写一个函数，生成配置文件prototxt
def Lenet(img_list, batch_size, include_acc = False):
    #第一层，数据输入层，以ImageData格式输入
    data, label = L.ImageData(source = img_list,batch_size = batch_size,ntop=2,root_folder = root,transform_param=dict(scale=0.00390625))
    #第二层，卷积层
    conv1 = L.Convolution(data,
                          convolution_param=dict(num_output=20,pad=0, stride=1, kernel_size=5, weight_filler=dict(type='xavier')))

    #第三层，池化层
    pool1 = L.Pooling(conv1, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride = 2))
    #卷积层
    conv2 = L.convolution(pool1,
                          convolution_param=dict(num_output = 50, pad = 0,kernel_size=5, stride = 1, weight_filler = dict(type = 'xavier')))
    #池化层
    pool2 = L.Pooling(conv2, pooling_param=dict(pool=P.Pooling.MAX,kernel_size=2, stride = 2))
    #全连接层
    fc3 = L.InnerProduct(pool2, inner_product_param=dict(num_output = 500, weight_filler = dict(type = 'xavier')))
    #激活函数层
    relu3 = L.ReLU(fc3, in_place = True)
    #全连层
    fc4 = L.InnerProduct(relu3, inner_product_param=dict(num_output = 10, weight_filler = dict(type = 'xavier')))
    #softMax层

    loss = L.SoftmaxWithLoss(fc4, label)
    if include_acc: #test阶段需要有accurary层
        acc = L.Accuracy(fc4, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)


def write_net():
    #写入train.prototxt
    print(1)
    with open(train_proto,'w') as f:
        f.write(str(Lenet(train_list, batch_size=64)))
    print(2)
    #写入test.prototxt
    with open(test_proto,'w') as f:
        f.write(str(Lenet(test_list, batch_size=100, include_acc=True)))


#编写一个函数生成参数文件

def gen_solver(solver_file, train_net, test_net):
    s = proto.caffe_pb2.SolverParameter() #保存生成参数文件
    s.train_net = train_net
    s.test_net.append(test_net)
    s.test_interval = 938 #60000/64，测试间隔参数：训练完一次所有的图片，进行一次测试
    s.test_iter.append(100) #10000/100 测试迭代次数，需要迭代100次，才完成一次所有数据的测试
    s.max_iter = 9380 #10 epochs, 938*10最大的迭代次数
    s.base_lr = 0.01 #基础学习率
    s.momentum = 0.9 #动量
    s.weight_decay = 0.0005 #权值衰减
    s.lr_policy = 'step' #学习率变化规则
    s.stepsize = 3000 #学习率变化频率
    s.gamma = 0.1 #学习率变化指数
    s.display = 20 #屏幕显示间隔
    s.snapshot = 938 #每938次迭代存储一次数据
    s.snapshot_prefix = root+'lenet' #caffemodel前缀
    s.type = 'SGD' #反向传播，随机梯度下降
    s.solver_mode = proto.caffe_pb2.SolverParameter.CPU
    #写入solver.prototxt
    with open(solver_file, 'w') as f:
        f.write(str(s))

def training(solver_proto):
    #caffe.set_device(0)
    #caffe.set_mode_cpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()

if __name__ == '__main__':
    write_net()
    gen_solver(solver_proto,train_proto,test_proto)
    training(solver_proto)