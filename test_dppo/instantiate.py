class ShortCutFlow(nn.Module):
    def __init__(self, 
                 network,
                 device,s
                 horizon_steps, 
                 action_dim, 
                 obs_dim, 
                 max_denoising_steps, 
                 cfg):
        