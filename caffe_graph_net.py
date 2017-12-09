#!/usr/bin/env python
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

class structtype:
    pass

def loadSolver(fn):
    with open(fn) as f:
        msg = caffe_pb2.SolverParameter()
        text_format.Merge(str(f.read()), msg)
    return msg

def loadNet(fn):
    with open(fn) as f:
        msg = caffe_pb2.NetParameter()
        text_format.Merge(str(f.read()), msg)
    return msg

def filterNetLayer(net, phase='TRAIN'):
    assert phase in ['TRAIN', 'TEST']
    if phase == 'TRAIN':
        phase = 'TEST'
    else:
        phase = 'TRAIN'

    net2 = structtype()
    net2.layer = [l for l in net.layer if phase not in unicode(l)]
    return net2

def graphNet(net, fn=None):
    from graphviz import Digraph

    g = Digraph(filename=fn)
    # layer node and blob node
    for l in net.layer:
        # layer node
        g.attr('node', shape='box')
        g.node('layer_' + l.name, label=l.name)
        # blob node
        g.attr('node', shape='ellipse')
        for t in l.top:
            g.node('blob_' + t, label=t)
        for b in l.bottom:
            g.node('blob_' + b, label=b)
    # edges
    for l in net.layer:
        name = 'layer_' + l.name
        for t in l.top:
            g.edge(name, 'blob_' + t)
        for b in l.bottom:
            g.edge('blob_' + b, name)
    return g

def graphNet2(net, fn=None):
    # blob as node, layer as edge
    from graphviz import Digraph

    g = Digraph(filename=fn)
    # layer node and blob node
    for l in net.layer:
        # blob node
        g.attr('node', shape='ellipse')
        for t in l.top:
            g.node('blob_' + t, label=t)
        for b in l.bottom:
            g.node('blob_' + b, label=b)
    # edges
    for l in net.layer:
        if len(l.top) == 0 or len(l.bottom) == 0:
            continue
        for t in l.top:
            for b in l.bottom:
                g.edge('blob_' + b, 'blob_' + t, l.name)

    return g

if __name__ == '__main__':
    import sys, os
    #net = loadNet('lenet_train_test.prototxt')
    #solver = loadSolver('lenet_solver.prototxt')

    net = loadNet(sys.argv[1])
    fn, ext = os.path.splitext(sys.argv[1])
    graphNet2(net, 'net.gv').render()
    os.rename('net.gv.pdf', fn + '.pdf')

#    train_net = filterNetLayer(net, 'TRAIN')
#    graphNet2(train_net, 'train_net.gv').render()
#    test_net = filterNetLayer(net, 'TEST')
#    graphNet2(test_net, 'test_net.gv').render()
