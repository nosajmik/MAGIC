#!/usr/bin/python3.7
"""
Borrowed from and rewritten based on Muhan's pytorch_DGCNN repo at
https://github.com/muhanzhang/pytorch_DGCNN
"""
import os
import sys
import glog as log
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' %
                os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB
from pytorch_util import weights_init, gnn_spmm


class DGCNN(nn.Module):
    def __init__(self, outputDim, numNodeFeats, numEdgeFeats=0,
                 latentDims=[32, 32, 32, 1], k=30,
                 poolingType='sort', endingLayers='conv1d',
                 conv2dChannel=64,
                 conv1dChannels=[16, 32],
                 conv1dKernSz=[0, 5], conv1dMaxPl=[2, 2]):
        """
        Args
            outputDim: dimension of the DGCNN. If equals zero, it will be
                       computed as the output of the final 1d conv layer;
                       Otherwise, an extra dense layer will be appended after
                       the final 1d conv layer to produce exact output size.
            numNodesFeats, numEdgeFeats: dim of the node/edge attributes.
            latend_dim: sizes of graph convolution layers.
            poolingType: type of pooling graph vertices, 'sort' or 'adaptive'.
            endingLayers: 'conv1d' or 'weight_vertices'.
                          NOT used if pooling layer is 'adaptive'.
            conv2dChannel: channel dimension of the 2d conv layer
                           before adaptive max pooling.
                           NOT used if pooling layer is 'sort'.
                           Default 64 to compatible with VGG11.
            conv1dChannels: channel dimension of the 2 conv1d layers
            conv1dKernSz: kernel size of the 2 1d conv layers.
                          conv1dKernSz[0] is manually set to sum(latentDims).
            conv1dMaxPl: maxpool kernel size and stride between 2 conv1d layers.
        """
        log.info('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latentDims = latentDims
        self.outputDim = outputDim
        self.k = k
        self.totalLatentDim = sum(latentDims)
        conv1dKernSz[0] = self.totalLatentDim

        self.graphConvParams = nn.ModuleList()
        self.graphConvParams.append(nn.Linear(numNodeFeats, latentDims[0]))
        for i in range(1, len(latentDims)):
            self.graphConvParams.append(
                nn.Linear(latentDims[i - 1], latentDims[i]))

        self.poolingType = poolingType
        if poolingType == 'adaptive':
            log.info(f'Unify graph sizes with ADAPTIVE pooling')
            self.conv2dParam = nn.Conv2d(in_channels=1,
                                         out_channels=conv2dChannel,
                                         kernel_size=13, stride=1, padding=6)
            self.adptPl = nn.AdaptiveMaxPool2d((self.k, self.totalLatentDim))

        weights_init(self)


    def forward(self, graphs, nodeFeats, edgeFeats):
        graphSizes = [graphs[i].num_nodes for i in range(len(graphs))]
        nodeDegs = [torch.Tensor(graphs[i].degs) + 1
                    for i in range(len(graphs))]
        nodeDegs = torch.cat(nodeDegs).unsqueeze(1)
        n2nSp, e2nSp, _ = S2VLIB.PrepareMeanField(graphs)
        if isinstance(nodeFeats, torch.cuda.FloatTensor):
            n2nSp = n2nSp.cuda()
            e2nSp = e2nSp.cuda()
            nodeDegs = nodeDegs.cuda()

        nodeFeats = Variable(nodeFeats)
        if edgeFeats is not None:
            edgeFeats = Variable(edgeFeats)

        n2nSp = Variable(n2nSp)
        e2nSp = Variable(e2nSp)
        nodeDegs = Variable(nodeDegs)

        convGraphs = self.graphConvLayers(nodeFeats, edgeFeats,
                                          n2nSp, e2nSp, graphSizes, nodeDegs)
        if self.poolingType == 'adaptive':
            return self.adptivePoolLayer(convGraphs, nodeFeats, graphSizes)


    def graphConvLayers(self, nodeFeats, edgeFeats, n2nSp, e2nSp,
                        graphSizes, nodeDegs):
        """graph convolution layers"""
        # if exists edge feature, concatenate to node feature vector
        if edgeFeats is not None:
            inputEdgeLinear = self.wE2L(edgeFeats)
            e2nPool = gnn_spmm(e2nSp, inputEdgeLinear)
            nodeFeats = torch.cat([nodeFeats, e2nPool], 1)

        lv = 0
        currMsgLayer = nodeFeats
        msgLayers = []
        while lv < len(self.latentDims):
            # Y = (A + I) * X
            n2npool = gnn_spmm(n2nSp, currMsgLayer) + currMsgLayer
            nodeLinear = self.graphConvParams[lv](n2npool)  # Y = Y * W
            normalizedLinear = nodeLinear.div(nodeDegs)  # Y = D^-1 * Y
            currMsgLayer = torch.tanh(normalizedLinear)
            msgLayers.append(currMsgLayer)
            lv += 1

        return torch.cat(msgLayers, 1)


    def adptivePoolLayer(self, convGraphs, nodeFeats, graphSizes):
        apGraphs = torch.zeros(len(graphSizes), self.conv2dParam.out_channels,
                               self.k, self.totalLatentDim)
        if isinstance(nodeFeats.data, torch.cuda.FloatTensor):
            apGraphs = apGraphs.cuda()

        graphIdx = 0
        for i in range(len(graphSizes)):
            graph = convGraphs[graphIdx: graphIdx + graphSizes[i]]
            # Convert to 4D matrix before applying conv2d
            toConv = graph.unsqueeze(0)
            toConv = toConv.unsqueeze(0)
            conved = self.conv2dParam(toConv)
            apGraphs[i] = self.adptPl(conved)
            graphIdx += graphSizes[i]

        return apGraphs
