import hydra
import os
from runners import runner

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg):
    if cfg.exp_name is None:
        cfg.exp_name = os.path.basename(__file__).rstrip(".py")
    
    return runner(cfg)

if __name__ == "__main__":
    main()
