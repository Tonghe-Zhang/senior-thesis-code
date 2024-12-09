"""
Parent pre-training agent class.

"""

import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
from copy import deepcopy

log = logging.getLogger(__name__)
from util.scheduler import CosineAnnealingWarmupRestarts

DEVICE = "cuda:0"

class ShortCutFlow():
    def __init__(self, num_channels, num_flows, num_layers, hidden_size, max_seq_len,
                 dropout=0.0, device="cuda"):
        super().__init__()
        self.train_cfg={ 
                'dataset_name': 'imagenet256',    # Environment name.
                'load_dir': None,                 # Logging dir (if not None, save params).
                'save_dir': None,                 # Logging dir (if not None, save params).
                'fid_stats': None,                # FID stats file.
                'seed': 10,                       # Random seed. Must be the same across all processes.
                'log_interval': 1000,             # Logging interval.
                'eval_interval': 20000,           # Eval interval.
                'save_interval': 100000,          # Save interval.
                'batch_size': 32,                 # Mini batch size.
                'max_steps': int(1_000_000),      # Number of training steps.
                'debug_overfit': 0,               # Debug overfitting.
                'mode': 'train',                  # train or inference.
            }
        self.model_cfg={
                'lr': 0.0001,
                'beta1': 0.9,
                'beta2': 0.999,
                'weight_decay': 0.1,
                'use_cosine': 0,
                'warmup': 0,
                'dropout': 0.0,
                'hidden_size': 64, # change this!
                'patch_size': 8, # change this!
                'depth': 2, # change this!
                'num_heads': 2, # change this!
                'mlp_ratio': 1, # change this!
                'class_dropout_prob': 0.1,
                'num_classes': 1000,
                'denoise_timesteps': 128,
                'cfg_scale': 4.0,
                'target_update_rate': 0.999,
                'use_ema': 0,
                'use_stable_vae': 1,
                'sharding': 'dp', # dp or fsdp.
                't_sampling': 'discrete-dt',
                'dt_sampling': 'uniform',
                'bootstrap_cfg': 0,
                'bootstrap_every': 8, # Make sure its a divisor of batch size.
                'bootstrap_ema': 1,
                'bootstrap_dt_bias': 0,
                'train_type': 'shortcut' # or naive.
            }
        
        self.network = None
        self.dropout_key=None
        
    def get_targets(self,images, labels, force_t, force_dt):
        """ 
        Receive data and labels, return labels, bootstrap targets, and shortcut configurations. 
        Inputs:
        FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1
        
        Outputs:
        x_t, v_t, t, dt_base, labels_dropped, info
        
        return the corrupted images x_t with their labels,
        bootstrapped targets and linear path v_t,
        number of time steps dt_base
        """
        label_key, time_key, noise_key = jax.random.split(key, 3)
        
        info = {}

        # 1) =========== Sample dt. ============
        bootstrap_batchsize = self.train_cfg['batch_size'] // self.model_cfg['bootstrap_every']
        log2_sections = np.log2(self.model_cfg['denoise_timesteps']).astype(np.int32)
        if self.model_cfg['bootstrap_dt_bias'] == 0:
            dt_base = np.repeat(log2_sections - 1 - np.arange(log2_sections), bootstrap_batchsize // log2_sections)
            dt_base = np.concatenate([dt_base, np.zeros(bootstrap_batchsize-dt_base.shape[0],)])
            num_dt_cfg = bootstrap_batchsize // log2_sections
        else:
            dt_base = np.repeat(log2_sections - 1 - np.arange(log2_sections-2), (bootstrap_batchsize // 2) // log2_sections)
            dt_base = np.concatenate([dt_base, np.ones(bootstrap_batchsize // 4), np.zeros(bootstrap_batchsize // 4)])
            dt_base = np.concatenate([dt_base, np.zeros(bootstrap_batchsize-dt_base.shape[0],)])
            num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections
        force_dt_vec = np.ones(bootstrap_batchsize, dtype=np.float32) * force_dt
        dt_base = np.where(force_dt_vec != -1, force_dt_vec, dt_base)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2

        # 2) =========== Sample t. ============
        dt_sections = np.power(2, dt_base) # [1, 2, 4, 8, 16, 32, ... M-1]
        t = np.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(np.float32)
        t = t / dt_sections # Between 0 and 1.
        force_t_vec = np.ones(bootstrap_batchsize, dtype=np.float32) * force_t
        t = np.where(force_t_vec != -1, force_t_vec, t)
        t_full = t[:, None, None, None]      # shape: (*, 1,1,1)

        # 3) =========== Generate Bootstrap Targets ============
        x_1 = images[:bootstrap_batchsize]
        x_0 = np.random.normal(noise_key, x_1.shape)
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        bst_labels = labels[:bootstrap_batchsize]
        call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema
        if not FLAGS.model['bootstrap_cfg']:
            v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False)
            t2 = t + dt_bootstrap
            
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = np.clip(x_t2, -4, 4)
            
            v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, bst_labels, train=False)
            v_target = (v_b1 + v_b2) / 2
        else:
            x_t_extra = np.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
            t_extra = np.concatenate([t, t[:num_dt_cfg]], axis=0)
            dt_base_extra = np.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
            labels_extra = np.concatenate([bst_labels, np.ones(num_dt_cfg, dtype=np.int32) * FLAGS.model['num_classes']], axis=0)
            v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
            v_b_cond = v_b1_raw[:x_1.shape[0]]
            v_b_uncond = v_b1_raw[x_1.shape[0]:]
            v_cfg = v_b_uncond + FLAGS.model['cfg_scale'] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
            v_b1 = np.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = np.clip(x_t2, -4, 4)
            x_t2_extra = np.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
            t2_extra = np.concatenate([t2, t2[:num_dt_cfg]], axis=0)
            v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
            v_b2_cond = v_b2_raw[:x_1.shape[0]]
            v_b2_uncond = v_b2_raw[x_1.shape[0]:]
            v_b2_cfg = v_b2_uncond + FLAGS.model['cfg_scale'] * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
            v_b2 = np.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)
            v_target = (v_b1 + v_b2) / 2

        v_target = np.clip(v_target, -4, 4)
        v_bst = v_target
        dtbase_bst = dt_base
        t_bst = t
        xt_bst = x_t
        l_bst = bst_labels

        # 4) =========== Generate Flow-Matching Targets ============
        labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
        labels_dropped = np.where(labels_dropout, FLAGS.model['num_classes'], labels)
        info['dropped_ratio'] = np.mean(labels_dropped == FLAGS.model['num_classes'])

        # Sample t.
        t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(np.float32)
        t /= FLAGS.model['denoise_timesteps']
        force_t_vec = np.ones(images.shape[0], dtype=np.float32) * force_t
        t = np.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
        t_full = t[:, None, None, None] # [batch, 1, 1, 1]

        # Sample flow pairs x_t, v_t.
        x_0 = jax.random.normal(noise_key, images.shape)
        x_1 = images
        x_t = x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = v_t = x_1 - (1 - 1e-5) * x_0
        dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(np.int32)
        dt_base_flow = np.ones(images.shape[0], dtype=np.int32) * dt_flow

        # bootstrap_batchsize = self.train_cfg['batch_size'] // self.model_cfg['bootstrap_every']
        bootstrap_batchsize = FLAGS.batch_size // FLAGS.model['bootstrap_every']
        flow_size = FLAGS.batch_size - bootstrap_batchsize  
        xt_flow=x_t[:flow_size]
        t_flow = t[:flow_size]
        dt_base_flow=dt_base_flow[:flow_size]
        v_flow=v_t[:flow_size]
        l_flow=labels_dropped[:flow_size]
        
        # ==== 5) Merge Flow+Bootstrap ====     
        x_t = np.concatenate([xt_bst, xt_flow], axis=0)
        t = np.concatenate([t_bst, t_flow], axis=0)
        dt_base = np.concatenate([dtbase_bst, dt_base_flow], axis=0)
        v_t = np.concatenate([v_bst, v_flow], axis=0)
        labels_dropped = np.concatenate([l_bst, l_flow], axis=0)
        
        info['bootstrap_ratio'] = np.mean(dt_base != dt_flow)
        info['v_magnitude_bootstrap'] = np.sqrt(np.mean(np.square(v_bst)))
        info['v_magnitude_b1'] = np.sqrt(np.mean(np.square(v_b1)))
        info['v_magnitude_b2'] = np.sqrt(np.mean(np.square(v_b2)))
        
        return x_t, v_t, t, dt_base, labels, info
    
    @torch.no_grad()
    def loss(self,images, labels, force_t, force_dt):
        
        # get labels / shortcut targets and shortcut schedules
        x_t, v_t, t, dt_base, labels, info = self.get_targets(images, labels, force_t, force_dt)

        # use neural network to predic the flow. 
        v_prime, logvars, activations = self.network(x_t, t, dt_base, labels, 
                                                     train=True, 
                                                     rngs={'dropout': self.dropout_key}, 
                                                     params=grad_params, 
                                                     return_activations=True)
        
        # calculate flow-matching loss defined as \ell_2 distance between prediction and labels / shortcut targets. 
        mse_v = np.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
        
        loss = np.mean(mse_v)

        # record relevant information
        info = {
            'loss': loss,
            'v_magnitude_prime': np.sqrt(np.mean(np.square(v_prime))),
            **{'activations/' + k : np.sqrt(np.mean(np.square(v))) for k, v in activations.items()},
        }

        if self.model_cfg['train_type'] in ['shortcut' , 'livereflow']:
            bootstrap_size = self.cfg.batch_size // self.model_cfg['bootstrap_every']
            info['loss_flow'] = np.mean(mse_v[bootstrap_size:])
            info['loss_bootstrap'] = np.mean(mse_v[:bootstrap_size])
        
        return loss, info
        