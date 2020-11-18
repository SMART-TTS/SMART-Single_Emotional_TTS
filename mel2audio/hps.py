class Hyperparameters():
    def __init__(self, args):
        self.n_ER_blocks = args.n_ER_blocks
        self.n_flow_blocks = args.n_flow_blocks
        self.n_layers = args.n_layers
        self.n_channels = args.n_channels
        self.hop_length = args.hop_length
        self.sqz_scale_i = args.sqz_scale_i
        self.sqz_scale = args.sqz_scale
        self.di_cycle = args.di_cycle
        self.pretrained = True if args.load_step > 0 else False