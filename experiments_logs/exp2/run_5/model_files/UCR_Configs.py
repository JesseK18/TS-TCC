class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = None #1 always
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 320 #320
        self.num_classes = None
        self.dropout = 0.35
        self.features_len = None
        self.device = 'cpu'
        # training configs
        self.num_epoch = 40

        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.001# 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 64

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 10
