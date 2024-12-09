# import torch

# # Assuming log2_sections and self.bootstrap_batchsize are defined
# log2_sections = 4  # Example value, replace with your actual value
# bootstrap_batchsize = 16  # Example value, replace with your actual value

# # Create the range tensor
# range_tensor = torch.arange(log2_sections)

# # Repeat the range tensor
# dt_base = (log2_sections - 1 - range_tensor).repeat_interleave(bootstrap_batchsize // log2_sections)

# # Concatenate with zeros to match the bootstrap_batchsize
# dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0])])

# print(dt_base)

import torch
a=torch.nn.MSELoss()
