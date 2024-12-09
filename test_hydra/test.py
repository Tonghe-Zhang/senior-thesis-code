
import os


import hydra
from omegaconf import OmegaConf
@hydra.main(version_base=None,
            config_path=os.path.join(os.getcwd(),'cfg'),   # can be override with --config-dir=
            config_name='test'                             # can be override with --config-name=
            )
def main(cfg:OmegaConf):
    OmegaConf.resolve(cfg)
    class_handle=hydra.utils.get_class(cfg._target_)
    
    my_model=hydra.utils.instantiate(cfg.model)
    my_model.foo()
    
    
    
    

if __name__ == '__main__':
    main()
    
    
    