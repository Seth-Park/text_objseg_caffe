from __future__ import absolute_import, division, print_function

import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P


###############################################################################
# Helper Methods
###############################################################################

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=mult)
    else:
        if finetune:
            mult = [dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)]
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=mult)
        else:
            mult = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
            filler = dict(type='xavier')
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad,
                                 param=mult, weight_filler=filler)
    return conv, L.ReLU(conv, in_place=True)


def conv(bottom, nout, ks=3, stride=1, pad=1, fix_param=False, finetune=False):
    if fix_param:
        mult = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=mult)
    else:
        if finetune:
            mult = [dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)]
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad, param=mult)
        else:
            mult = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
            filler = dict(type='xavier')
            conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                 num_output=nout, pad=pad,
                                 param=mult, weight_filler=filler)
    return conv


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


################################################################################
# Model Generation
###############################################################################

def generate_model(split, config):
    n = caffe.NetSpec()
    batch_size = config.N
    mode_str = str(dict(split=split, batch_size=batch_size))
    n.language, n.cont, n.image, n.spatial, n.label = L.Python(module=config.data_provider,
                                                               layer=config.data_provider_layer,
                                                               param_str=mode_str,
                                                               ntop=5)

    # fully conv
    n.fcn_fc6, n.fcn_relu6 = conv_relu(n.image, 2048, ks=7, pad=3)
    if config.fcn_dropout:
        n.fcn_drop6 = L.Dropout(n.fcn_relu6, dropout_ratio=0.5, in_place=True)
        n.fcn_fc7, n.fcn_relu7 = conv_relu(n.fcn_drop6, 4096, ks=1, pad=0)
        n.fcn_drop7 = L.Dropout(n.fcn_relu7, dropout_ratio=0.5, in_place=True)
        n.fcn_fc8 = conv(n.fcn_drop7, 1000, ks=1, pad=0)
    else:
        n.fcn_fc7, n.fcn_relu7 = conv_relu(n.fcn_relu6, 4096, ks=1, pad=0)
        n.fcn_fc8 = conv(n.fcn_relu7, 1000, ks=1, pad=0)

    
    # embedding
    n.embed = L.Embed(n.language, input_dim=config.vocab_size,
                      num_output=config.embed_dim,
                      weight_filler=dict(type='uniform', min=-0.08, max=0.08))

    # LSTM
    n.lstm = L.LSTM(n.embed, n.cont,
                    recurrent_param=dict(num_output=config.lstm_dim,
                                         weight_filler=dict(type='uniform', min=-0.08, max=0.08),
                                         bias_filler=dict(type='constant', value=0)))
    tops = L.Slice(n.lstm, ntop=config.T, slice_param=dict(axis=0))
    for i in range(config.T - 1):
        n.__setattr__('slice'+str(i), tops[i])
        n.__setattr__('silence'+str(i), L.Silence(tops[i], ntop=0))
    n.lstm_out = tops[-1]
    n.lstm_feat = L.Reshape(n.lstm_out, reshape_param=dict(shape=dict(dim=[-1, config.lstm_dim])))

    # L2 normalize and Tile LSTM feature
    n.lstm_l2norm = L.L2Normalize(n.lstm_feat)
    n.lstm_resh = L.Reshape(n.lstm_l2norm,
                            reshape_param=dict(shape=dict(dim=[-1, config.lstm_dim, 1, 1])))
    n.lstm_tile_1 = L.Tile(n.lstm_resh, axis=2, tiles=config.featmap_H)
    n.lstm_tile_2 = L.Tile(n.lstm_tile_1, axis=3, tiles=config.featmap_W)

    # L2 Normalize image features
    n.img_l2norm = L.L2Normalize(n.fcn_fc8)

    # Concatenate
    n.feat_all = L.Concat(n.lstm_tile_2, n.img_l2norm, n.spatial, concat_param=dict(axis=1))

    # MLP Classifier over concatenated feature
    n.fcn_l1, n.fcn_relu1 = conv_relu(n.feat_all, config.mlp_hidden_dims, ks=1, pad=0)
    if config.mlp_dropout:
        n.fcn_drop1 = L.Dropout(n.fcn_relu1, dropout_ratio=0.5, in_place=True)
        n.fcn_scores = conv(n.fcn_drop1, 1, ks=1, pad=0)
    else:
        n.fcn_scores = conv(n.fcn_relu1, 1, ks=1, pad=0)
    
    # Loss Layer
    n.loss = L.SigmoidCrossEntropyLoss(n.fcn_scores, n.label)

    return n.to_proto()



