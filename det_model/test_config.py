# Model Params
T = 20
N = 100 
input_H = 224 
input_W = 224 
featmap_H = (input_H // 32)
featmap_W = (input_W // 32)
vocab_size = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

D_im = 1000
D_text = lstm_dim


# Testing Params
gpu_id = 0
pretrained_model = '/data2/seth/nl_seg_caffe/snapshots/det/_iter_25000.caffemodel'

correct_iou_thresh = 0.5
use_nms = False
nms_thresh = 0.3


# Data Params
data_provider = 'referit_data_provider'
data_provider_layer = 'ReferitDataProviderLayer'

image_dir = './referit/referit-dataset/images/'
bbox_proposal_dir = './referit/data/referit_edgeboxes_top100/'
query_file = './referit/data/referit_query_test.json'
bbox_file = './referit/data/referit_bbox.json'
imcrop_file = './referit/data/referit_imcrop.json'
imsize_file = './referit/data/referit_imsize.json'
vocab_file = './referit/data/vocabulary_referit.txt'


