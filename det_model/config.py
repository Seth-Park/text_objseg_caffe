# Model Params
T = 20
N = 50
input_H = 224 
input_W = 224 
featmap_H = (input_H // 32)
featmap_W = (input_W // 32)
vocab_size = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Training Params
gpu_id = 0
max_iter = 25000

weights = '/data3/seth/VGG_ILSVRC_16_layers.caffemodel'
fix_vgg = True  # set False to finetune VGG net
vgg_dropout = False
mlp_dropout = False

# Data Params
data_provider = 'referit_data_provider'
data_provider_layer = 'ReferitDataProviderLayer'

data_folder = '/mnt/coelacanth/seth/data/exp-referit/data/train_batch_det/'
data_prefix = 'referit_train_det'


