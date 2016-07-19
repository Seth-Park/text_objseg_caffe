# Model Params
T = 20
N = 1
input_H = 512 
input_W = 512 
featmap_H = (input_H // 32)
featmap_W = (input_W // 32)
vocab_size = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500


# Testing Params
gpu_id = 2 
pretrained_model = '/data2/seth/nl_seg_caffe/snapshots/high_res_seg/_iter_25000.caffemodel'

score_thresh = 1e-9


# Data Params
data_provider = 'referit_data_provider'
data_provider_layer = 'ReferitDataProviderLayer'

image_dir = './referit/referit-dataset/images/'
mask_dir = './referit/referit-dataset/mask/'
query_file = './referit/data/referit_query_test.json'
bbox_file = './referit/data/referit_bbox.json'
imcrop_file = './referit/data/referit_imcrop.json'
imsize_file = './referit/data/referit_imsize.json'
vocab_file = './referit/data/vocabulary_referit.txt'


