class Config():
    def __init__(self):
        # Model Params
        self.T = 20
        self.N = 100 
        self.input_H = 224 
        self.input_W = 224 
        self.featmap_H = (self.input_H // 32)
        self.featmap_W = (self.input_W // 32)
        self.vocab_size = 8803
        self.embed_dim = 1000
        self.lstm_dim = 1000
        self.mlp_hidden_dims = 500
        self.fix_vgg = True
        self.vgg_dropout = False
        self.mlp_dropout = False

        self.D_im = 1000
        self.D_text = self.lstm_dim


        # Testing Params
        self.gpu_id = 1
        self.pretrained_model = '/x/dhpseth/text_objseg_caffe/snapshots/det/_iter_25000.caffemodel'

        self.correct_iou_thresh = 0.5
        self.use_nms = False
        self.nms_thresh = 0.3


        # Data Params
        self.data_provider = 'referit_data_provider'
        self.data_provider_layer = 'ReferitDataProviderLayer'

        self.image_dir = './referit/referit-dataset/images/'
        self.bbox_proposal_dir = './referit/data/referit_edgeboxes_top100/'
        self.query_file = './referit/data/referit_query_test.json'
        self.bbox_file = './referit/data/referit_bbox.json'
        self.imcrop_file = './referit/data/referit_imcrop.json'
        self.imsize_file = './referit/data/referit_imsize.json'
        self.vocab_file = './referit/data/vocabulary_referit.txt'


