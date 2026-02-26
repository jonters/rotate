import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os, sys
# Ensure the project root (parent of marl/) is on sys.path so that local
# packages like `envs` are found even after Hydra changes the working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

@hydra.main(version_base=None, config_path="configs", config_name="heldout_ego")
def main(cfg):
    '''Run evaluation. 
    All evaluators assume that the path to the ego agent is provided at config["ego_agent"]["path"]
    and that all information necessary to properly initialize the ego agent is provided at config["ego_agent"]
    '''
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "regret" in cfg["name"]:
        from regret_evaluator import run_regret_evaluation
        run_regret_evaluation(cfg)

    elif "heldout_ego" in cfg["name"]:
        from heldout_evaluator import run_heldout_evaluation
        run_heldout_evaluation(cfg, print_metrics=True)

    elif "heldout_xp" in cfg["name"]:
        from evaluation.generate_xp_matrix import run_heldout_xp_evaluation
        run_heldout_xp_evaluation(cfg, print_metrics=True)
    elif "human_proxy_eval" in cfg["name"]:
        from human_proxy_evaluator import run_human_proxy_evaluation
        run_human_proxy_evaluation(cfg, print_metrics=True)

    else: 
        raise ValueError(f"Evaluator {cfg['name']} not found.")

if __name__ == '__main__':
    main()

"""
cd /scratch/cluster/jyliu/Documents/rotate/rotate/evaluation

python evaluation/run.py \
  task=lbf-fov-3 \
  name="lbf-fov-3/heldout_ego_eval" \
  ego_agent.path="results/lbf-fov-3/rotate/default_label/2026-02-25_15-10-24/saved_train_run" \
  ego_agent.actor_type=s5 \
  ego_agent.test_mode=true \
  ego_agent.idx_list=[[0,-1]] \
  +ego_agent.S5_D_MODEL=64 \
  +ego_agent.S5_SSM_SIZE=64 \
  +ego_agent.S5_ACTOR_CRITIC_HIDDEN_DIM=256 \
  +ego_agent.FC_N_LAYERS=3 \
  +ego_agent.custom_loader.name=open_ended \
  +ego_agent.custom_loader.type=ego
"""