import argparse
import json
from pathlib import Path
import os
from overcooked_ai_py.agents.agent import PlanningAgent
from overcooked_ai_py.planning.planners import MediumLevelActionManager, COUNTERS_MLG_PARAMS, MotionPlanner
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from multiprocessing import Pool

class Mlam_generator():
    def __init__(self, config):
        self.config = config
    def gen_mlam(self, layout_name):
        mdp = OvercookedGridworld.from_layout_name(layout_name, self.config.get("layouts_dir", "overcooked_ai_py/data/layouts"))
        counter_params = COUNTERS_MLG_PARAMS
        if mdp.counter_goals:
            counter_params["counter_goals"] = mdp.counter_goals
            counter_params["counter_drop"] = mdp.counter_goals
            counter_params["counter_pickup"] = mdp.counter_goals
        mlam = MediumLevelActionManager.from_pickle_or_compute(
                mdp, counter_params, force_compute=False, info=True)
        motion_planner = MotionPlanner.from_pickle_or_compute(mdp, counter_params, force_compute=False, info=True)
        if self.config["copy"]:
            Path("/tmp_user/ldtis960b/mleguill/pickles/"+self.config["name"]).mkdir(parents=True, exist_ok=True)
            motion_planner.save_to_file("/tmp_user/ldtis960b/mleguill/pickles/"+os.path.join(self.config["name"], layout_name+"_mp.pkl"))
            mlam.save_to_file("/tmp_user/ldtis960b/mleguill/pickles/"+os.path.join(self.config["name"], layout_name+"_am.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--copy", action="store_true", default=False)
    args=parser.parse_args()
    with open("./config.json", 'r') as f:
        CONFIG = json.load(f)
    config = CONFIG[args.config]
    config["copy"] = args.copy
    config["name"] = args.config
    mlam_generator = Mlam_generator(config)
    all_layouts = []
    for key, bloc in config["blocs"].items():
        for layout in bloc:
            all_layouts.append(layout)
    print(all_layouts)
    all_layouts = list(set(all_layouts))
    with Pool() as pool:
        pool.map(mlam_generator.gen_mlam, all_layouts)






    





