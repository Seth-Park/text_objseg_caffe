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

def inference():
    with open('./fc8.prototxt', 'w') as f:
        f.write(str(det_model.generate_fc8('val', test_config.N)))
    with open('./scores.prototxt', 'w') as f:
        f.write(str(det_model.generate_scores('val', test_config.N)))

    caffe.set_device(test_config.gpu_id)
    caffe.set_mode_gpu()

    # Load pretrained model
    fc8_net = caffe.Net('./fc8.prototxt',
                        test_config.pretrained_model,
                        caffe.TEST)

    scores_net = caffe.Net('./scores.prototxt',
                           test_config.pretrained_model,
                           caffe.TEST)

    ################################################################################
    # Load annotations and bounding box proposals
    ################################################################################

    query_dict = json.load(open(test_config.query_file))
    bbox_dict = json.load(open(test_config.bbox_file))
    imcrop_dict = json.load(open(test_config.imcrop_file))
    imsize_dict = json.load(open(test_config.imsize_file))
    imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
    vocab_dict = text_processing.load_vocab_dict_from_file(test_config.vocab_file)

    # Object proposals
    bbox_proposal_dict = {}
    for imname in imlist:
        bboxes = np.loadtxt(test_config.bbox_proposal_dir + imname[:-4] + '.txt').astype(int).reshape((-1, 4))
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
    imcrop_val = np.zeros((test_config.N, test_config.input_H, test_config.input_W, 3), dtype=np.float32)
    spatial_val = np.zeros((test_config.N, 8), dtype=np.float32)
    text_seq_val = np.zeros((test_config.T, test_config.N), dtype=np.int32)

    dummy_text_seq = np.zeros((test_config.T, test_config.N), dtype=np.int32)
    dummy_cont = np.zeros((test_config.T, test_config.N), dtype=np.int32)
    dummy_label = np.zeros((test_config.N, 1))

    num_im = len(imlist)
    for n_im in tqdm(range(num_im)):
        imname = imlist[n_im]
        imsize = imsize_dict[imname]
        bbox_proposals = bbox_proposal_dict[imname]
        num_proposal = bbox_proposals.shape[0]
        assert(test_config.N >= num_proposal)

        # Extract visual features from all proposals
        im = skimage.io.imread(test_config.image_dir + imname)
        if im.ndim == 2:
            im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
        imcrop_val[:num_proposal, ...] = im_processing.crop_bboxes_subtract_mean(
            im, bbox_proposals, test_config.input_H, det_model.channel_mean)
        imcrop_val_trans = imcrop_val.transpose((0, 3, 1, 2))

        # Extract bounding box features from proposals
        spatial_val[:num_proposal, ...] = \
            processing_tools.spatial_feature_from_bbox(bbox_proposals, imsize)

        fc8_net.blobs['language'].data[...] = dummy_text_seq
        fc8_net.blobs['cont'].data[...] = dummy_cont
        fc8_net.blobs['image'].data[...] = imcrop_val_trans
        fc8_net.blobs['spatial'].data[...] = spatial_val
        fc8_net.blobs['label'].data[...] = dummy_label

        fc8_net.forward()
        fc8_val = fc8_net.blobs['fc8'].data[...].copy()

        # Extract textual features from sentences
        for imcrop_name, gt_bbox, description in flat_query_dict[imname]:
            proposal_IoUs = eval_tools.compute_bbox_iou(bbox_proposals, gt_bbox)

            # Extract language feature
            text = text_processing.preprocess_sentence(description, vocab_dict, test_config.T)
            text_seq_val[...] = np.array(text, dtype=np.int32).reshape((-1, 1))

            cont_val = text_processing.create_cont(text_seq_val)

            scores_net.blobs['language'].data[...] = text_seq_val
            scores_net.blobs['cont'].data[...] = cont_val
            scores_net.blobs['img_feature'].data[...] = fc8_val
            scores_net.blobs['spatial'].data[...] = spatial_val
            scores_net.blobs['label'].data[...] = dummy_label

            scores_net.forward()

            scores_val = scores_net.blobs['scores'].data.copy()
            scores_val = scores_val[:num_proposal, ...].reshape(-1)

            # Sort the scores for the proposals
            if test_config.use_nms:
                top_ids = eval_tools.nms(proposal.astype(np.float32), scores_val, test_config.nms_thresh)
            else:
                top_ids = np.argsort(scores_val)[::-1]

            # Evaluate on bounding boxes
            for n_eval_num in range(len(eval_bbox_num_list)):
                eval_bbox_num = eval_bbox_num_list[n_eval_num]
                bbox_correct[n_eval_num] += \
                    np.any(proposal_IoUs[top_ids[:eval_bbox_num]] >= test_config.correct_iou_thresh)
            bbox_total += 1

    print('Final results on the whole test set')
    result_str = ''
    for n_eval_num in range(len(eval_bbox_num_list)):
        result_str += 'recall@%s = %f\n' % \
            (str(eval_bbox_num_list[n_eval_num]), bbox_correct[n_eval_num]/bbox_total)
    print(result_str)

if __name__ == '__main__':
    inference()
