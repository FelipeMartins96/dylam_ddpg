import hydra
import os
from runners import runner

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg):
    if cfg.experiment.exp_name is None:
        cfg.experiment.exp_name = os.path.basename(__file__).rstrip(".py")
    
    return runner(cfg.experiment)

if __name__ == "__main__":
    main()
