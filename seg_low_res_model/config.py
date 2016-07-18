# Model Params
T = 20
N = 10 
input_H = 512
input_W = 512
featmap_H = (input_H // 32)
featmap_W = (input_W // 32)
vocab_size = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Training Params
gpu_id = 0
max_iter = 30000

weights = '/data2/seth/nl_seg_caffe/snapshots/det/_iter_25000.caffemodel'
fix_vgg = True 
vgg_dropout = False
mlp_dropout = False

# Data Params
data_provider = 'referit_data_provider'
data_provider_layer = 'ReferitDataProviderLayer'

data_folder = '/mnt/coelacanth/seth/data/exp-referit/data/train_batch_seg/'
data_prefix = 'referit_train_seg'


