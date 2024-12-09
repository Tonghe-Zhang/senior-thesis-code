Pre-training command using hopper without wandb login. 

```python
python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/gym/pretrain/hopper-medium-v2 wandb=null
```



configuration information

```yaml
horizon_steps: 4
action_dim: 3

obs_dim: 11
cond_steps: 1

train:
  batch_size: 128
  
  n_epochs: 3000
```



`self.dataloader_train`:

​	length == 7761. 

`batch_train`: a named tuple.     Batch(actions=…, condition=…) 

​	`batch_train.actions` 	[128,4,3] Tensor.     

​							   batch_size x horizon_steps x action_dim

​	`batch_train.conditions`   dict

​		 `batch_train.conditions['state']`    [128,1,11] Tensor    

​							   batch_size  x cond_steps  x obs_dim





























The dataset provided in the code snippet is a pre-training dataset for a diffusion policy. It is loaded from a file specified by dataset_path, which can be either a .npz (NumPy zip) or .pkl (Python pickle) file. The dataset contains arrays of states, actions, and optionally images, along with arrays of trajectory lengths and rewards/dones. The dataset is loaded onto the specified device (e.g., "cuda:0") for efficient computation. The states and actions are converted to torch tensors of type float, while the images (if present) are converted to torch tensors. The following parameters are used to configure the dataset:

horizon_steps: This parameter specifies the number of time steps to consider for each sample in the dataset. It determines the length of the sequence of actions and conditions to be used for training the diffusion policy.

cond_steps: This parameter specifies the number of previous states (proprioceptive, etc.) to include in the conditioning information for each sample. It determines the history of states to be used for training the policy.

img_cond_steps: This parameter specifies the number of previous images to include in the conditioning information for each sample. It determines the history of images to be used for training the policy.

max_n_episodes: This parameter specifies the maximum number of episodes to load from the dataset. It helps control the size of the dataset used for training.

use_img: This parameter specifies whether to use images in the dataset for training. If set to True, the images are included in the conditioning information for each sample.

device: This parameter specifies the device (e.g., CPU or GPU) to load the dataset onto.

In the context of training a diffusion policy using this dataset, horizon_steps determines the length of the sequence of actions and conditions used for training. cond_steps and img_cond_steps specify the history of states and images to be used for training the policy. The dataset is loaded onto the specified device for efficient computation. Please note that the code snippet provided is a part of a larger class, and the specific implementation details may vary depending on the requirements of your training pipeline.