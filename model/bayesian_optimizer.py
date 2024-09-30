from bayes_opt import BayesianOptimization, UtilityFunction
from data.data_utils import format_points
import numpy as np
import logging


class BayesianOptimizer:
    def __init__(self, args):
        pbounds = {
            'x1': (0, 2.99),  # timbre
            'x2': (0, 4.99),  # note
            'x3': (0, 2.99),  # volume
            'x4': (0, 2.99),  # pattern
            'x5': (0, 2.99),  # duration
            'x6': (0, 4.99),  # intensity
        }
        self.bo = BayesianOptimization(f=None, pbounds=pbounds, verbose=2, random_state=1)
        self.utility = UtilityFunction(kind="ucb", kappa=10)
        self.current_params = None

    def run(self, trial_idx):
        next_point_to_probe = self.bo.suggest(self.utility)
        self.current_params = next_point_to_probe

        points = [
            int(np.floor(next_point_to_probe['x1'])),
            int(np.floor(next_point_to_probe['x2'])),
            int(np.floor(next_point_to_probe['x3'])),
            int(np.floor(next_point_to_probe['x4'])),
            int(np.floor(next_point_to_probe['x5'])),
            int(np.floor(next_point_to_probe['x6'])),
        ]

        logging.info(format_points(points))
        with open(f"{self.args.baselog_dir}/points.txt", 'a') as f:
            f.write(f'Trial_{trial_idx}_{format_points(points)}\n')
        
        return points

    def register_result(self, target):
        if self.current_params:
            self.bo.register(params=self.current_params, target=target)
