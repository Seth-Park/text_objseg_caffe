from __future__ import absolute_import, division, print_function

import sys
import os;
import skimage.io
import numpy as np
import caffe
import json
from tqdm import tqdm

import det_model
import test_config

from util import processing_tools, im_processing, text_processing, eval_tools

################################################################################
# Evaluation network
################################################################################

def inference(config):
    with open('./det_model/test.prototxt', 'w') as f:
        f.write(str(det_model.generate_model('val', config)))

    caffe.set_device(config.gpu_id)
    caffe.set_mode_gpu()

    # Load pretrained model
    net = caffe.Net('./det_model/test.prototxt',
                    config.pretrained_model,
                    caffe.TEST)

    # Load ResNet
    resnet = caffe.Net(config.resnet_prototxt,
                       config.resnet_caffemodel,
                       caffe.TEST)

    # ResNet mean
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(config.resnet_mean_path, 'rb').read()
    blob.ParseFromString(data)
    resnet_mean = np.array(caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3, 224, 224)
    resnet_mean = resnet_mean.transpose((1,2,0))


    ################################################################################
    # Load annotations and bounding box proposals
    ################################################################################

    query_dict = json.load(open(config.query_file))
    bbox_dict = json.load(open(config.bbox_file))
    imcrop_dict = json.load(open(config.imcrop_file))
    imsize_dict = json.load(open(config.imsize_file))
    imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
    vocab_dict = text_processing.load_vocab_dict_from_file(config.vocab_file)

    # Object proposals
    bbox_proposal_dict = {}
    for imname in imlist:
        bboxes = np.loadtxt(config.bbox_proposal_dir + imname[:-4] + '.txt').astype(int).reshape((-1, 4))
        bbox_proposal_dict[imname] = bboxes

    ################################################################################
    # Flatten the annotations
    ################################################################################

    flat_query_dict = {imname: [] for imname in imlist}
    for imname in imlist:
        this_imcrop_names = imcrop_dict[imname]
        for imcrop_name in this_imcrop_names:
            gt_bbox = bbox_dict[imcrop_name]
            if imcrop_name not in query_dict:
                continue
            this_descriptions = query_dict[imcrop_name]
            for description in this_descriptions:
                flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

    ################################################################################
    # Testing
    ################################################################################

    eval_bbox_num_list = [1, 10, 100]
    bbox_correct = np.zeros(len(eval_bbox_num_list), dtype=np.int32)
    bbox_total = 0

    # Pre-allocate arrays
    imcrop_val = np.zeros((config.N, config.input_H, config.input_W, 3), dtype=np.float32)
    spatial_val = np.zeros((config.N, 8), dtype=np.float32)
    text_seq_val = np.zeros((config.T, config.N), dtype=np.int32)
    dummy_label = np.zeros((config.N, 1))

    num_im = len(imlist)
    for n_im in tqdm(range(num_im)):
        imname = imlist[n_im]
        imsize = imsize_dict[imname]
        bbox_proposals = bbox_proposal_dict[imname]
        num_proposal = bbox_proposals.shape[0]
        assert(config.N >= num_proposal)

        # Extract visual features from all proposals
        im = skimage.io.imread(config.image_dir + imname)
        if im.ndim == 2:
            im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
        imcrop_val[:num_proposal, ...] = im_processing.crop_bboxes_subtract_mean(
            im, bbox_proposals, config.input_H, resnet_mean)
        imcrop_val_trans = imcrop_val.transpose((0, 3, 1, 2))

        # Extract bounding box features from proposals
        spatial_val[:num_proposal, ...] = \
            processing_tools.spatial_feature_from_bbox(bbox_proposals, imsize)

        total_blob = []
        resnet.blobs['data'].data[...] = imcrop_val_trans[:25]
        resnet.forward()
        total_blob.append(resnet.blobs['res5c'].data[...].copy())
        resnet.blobs['data'].data[...] = imcrop_val_trans[25:50]
        resnet.forward()
        total_blob.append(resnet.blobs['res5c'].data[...].copy())
        resnet.blobs['data'].data[...] = imcrop_val_trans[50:75]
        resnet.forward()
        total_blob.append(resnet.blobs['res5c'].data[...].copy())
        resnet.blobs['data'].data[...] = imcrop_val_trans[75:]
        resnet.forward()
        total_blob.append(resnet.blobs['res5c'].data[...].copy())
        feature = np.concatenate(total_blob)

        # Extract textual features from sentences
        for imcrop_name, gt_bbox, description in flat_query_dict[imname]:
            proposal_IoUs = eval_tools.compute_bbox_iou(bbox_proposals, gt_bbox)

            # Extract language feature
            text = text_processing.preprocess_sentence(description, vocab_dict, config.T)
            text_seq_val[...] = np.array(text, dtype=np.int32).reshape((-1, 1))

            cont_val = text_processing.create_cont(text_seq_val)

            net.blobs['language'].data[...] = text_seq_val
            net.blobs['cont'].data[...] = cont_val
            net.blobs['image'].data[...] = feature
            net.blobs['spatial'].data[...] = spatial_val
            net.blobs['label'].data[...] = dummy_label

            net.forward()

            scores_val = net.blobs['scores'].data.copy()
            scores_val = scores_val[:num_proposal, ...].reshape(-1)

            # Sort the scores for the proposals
            if config.use_nms:
                top_ids = eval_tools.nms(proposal.astype(np.float32), scores_val, config.nms_thresh)
            else:
                top_ids = np.argsort(scores_val)[::-1]

            # Evaluate on bounding boxes
            for n_eval_num in range(len(eval_bbox_num_list)):
                eval_bbox_num = eval_bbox_num_list[n_eval_num]
                bbox_correct[n_eval_num] += \
                    np.any(proposal_IoUs[top_ids[:eval_bbox_num]] >= config.correct_iou_thresh)
            bbox_total += 1

    print('Final results on the whole test set')
    result_str = ''
    for n_eval_num in range(len(eval_bbox_num_list)):
        result_str += 'recall@%s = %f\n' % \
            (str(eval_bbox_num_list[n_eval_num]), bbox_correct[n_eval_num]/bbox_total)
    print(result_str)

if __name__ == '__main__':
    config = test_config.Config()
    inference(config)
