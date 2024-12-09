import hydra
from omegaconf import OmegaConf
import os



""" 
python script/run.py --config-path cfg/gym/pretrain/halfcheetah-medium-v2 --config-name pre_shortcutflow_mlp.yaml
"""

@hydra.main(version_base=None,
            config_path=os.path.join(os.getcwd(),'cfg'),   # can be override with --config-dir=
            config_name='test'                             # can be override with --config-name=
            )
def main(cfg:OmegaConf):
    OmegaConf.resolve(cfg)
    class_handle=hydra.utils.get_class(cfg._target_)
    
    cls=class_handle(cfg)
    agent=cls(cfg)
    agent.run()

if __name__ == '__main__':
    main()
    