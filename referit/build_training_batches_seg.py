from __future__ import absolute_import, division, print_function

import numpy as np
import os
import json
import skimage
import skimage.io
import skimage.transform
import cv2
import caffe

from util import im_processing, text_processing
from util.io import load_referit_gt_mask as load_gt_mask

################################################################################
# Parameters
################################################################################

image_dir = './referit/referit-dataset/images/'
mask_dir = './referit/referit-dataset/mask/'
query_file = './referit/data/referit_query_trainval.json'
imsize_file = './referit/data/referit_imsize.json'
vocab_file = './referit/data/vocabulary_referit.txt'

# Saving directory
data_folder = './referit/data/train_batch_seg_resnet/'
data_prefix = 'referit_train_seg'

# Model Params
T = 20
N = 10
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)

################################################################################
# Load annotations
################################################################################

query_dict = json.load(open(query_file))
imsize_dict = json.load(open(imsize_file))
imcrop_list = query_dict.keys()
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Collect training samples
################################################################################

# Initialize ResNet-152
gpu_id = 0
resnet_prototxt_path = '/home/dhpseth/vqa/00_data_preprocess/ResNet-152-512-deploy.prototxt'
resnet_mean_path = '/home/dhpseth/vqa/00_data_preprocess/ResNet_mean.binaryproto'
resnet_caffemodel_path = '/data3/seth/ResNet-152-model.caffemodel'
extract_layer = 'res5c'
extract_layer_size = (2048, 16, 16)

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

resnet = caffe.Net(resnet_prototxt_path, resnet_caffemodel_path, caffe.TEST)

# mean subtraction
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(resnet_mean_path, 'rb').read()
blob.ParseFromString(data)
resnet_mean = np.array(caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3, 224, 224)
resnet_mean = cv2.resize(resnet_mean.transpose((1,2,0)), (input_H, input_W))
resnet_mean = resnet_mean.transpose((2,0,1))


training_samples = []
num_imcrop = len(imcrop_list)
for n_imcrop in range(num_imcrop):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop+1, num_imcrop))
    imcrop_name = imcrop_list[n_imcrop]

    # Image and mask
    imname = imcrop_name.split('_', 1)[0] + '.jpg'
    mask_name = imcrop_name + '.mat'
    im = skimage.io.imread(image_dir + imname)
    mask = load_gt_mask(mask_dir + mask_name).astype(np.float32)

    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = processed_im[:, :, np.newaxis]

    # ResNet feature extraction
    im_preprocessed = processed_im.astype(np.float32).transpose((2, 0, 1)) - resnet_mean
    resnet.blobs['data'].data[0,...] = im_preprocessed
    resnet.forward()
    feature = resnet.blobs[extract_layer].data[0].copy()
    feature = feature.reshape(extract_layer_size)

    processed_mask = im_processing.resize_and_pad(mask, input_H, input_W)
    subsampled_mask = skimage.transform.downscale_local_mean(processed_mask, (32, 32))

    labels_fine = (processed_mask > 0)
    labels_coarse = (subsampled_mask > 0)

    for description in query_dict[imcrop_name]:
        text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
        training_samples.append((processed_im, feature, text_seq, labels_coarse, labels_fine))

# Shuffle the training instances
np.random.seed(3)
shuffle_idx = np.random.permutation(len(training_samples))
shuffled_training_samples = [training_samples[n] for n in shuffle_idx]
print('total training instance number: %d' % len(shuffled_training_samples))

# Create training batches
num_batch = len(shuffled_training_samples) // N
print('total batch number: %d' % num_batch)

################################################################################
# Save training samples to disk
################################################################################

text_seq_batch = np.zeros((T, N), dtype=np.int32)
imcrop_batch = np.zeros((N, input_H, input_W, 3), dtype=np.uint8)
feature_batch = np.zeros((N, 2048, 16, 16), dtype=np.float32)
label_coarse_batch = np.zeros((N, featmap_H, featmap_W, 1), dtype=np.bool)
label_fine_batch = np.zeros((N, input_H, input_W, 1), dtype=np.bool)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        processed_im, feature, text_seq, labels_coarse, labels_fine = shuffled_training_samples[n_sample]
        text_seq_batch[:, n_sample-batch_begin] = text_seq
        imcrop_batch[n_sample-batch_begin, ...] = processed_im
        feature_batch[n_sample-batch_begin, ...] = feature
        label_coarse_batch[n_sample-batch_begin, ...] = labels_coarse[:, :, np.newaxis]
        label_fine_batch[n_sample-batch_begin, ...] = labels_fine[:, :, np.newaxis]

    cont_batch = text_processing.create_cont(text_seq_batch)
    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
         text_seq_batch=text_seq_batch,
         cont_batch=cont_batch,
         imcrop_batch=imcrop_batch,
         feature_batch=feature_batch,
         label_coarse_batch=label_coarse_batch,
         label_fine_batch=label_fine_batch)
