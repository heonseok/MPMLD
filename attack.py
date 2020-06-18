import os

import numpy as np
from scipy.stats import entropy

from utils import classify_membership


class Attacker(object):
    def __init__(self, args):
        self.cls_name = os.path.join('{}_setsize{}'.format(args.model_type, args.setsize),
                                     'repeat{}'.format(args.repeat_idx))
        self.cls_path = os.path.join(args.base_path, 'classifier', self.cls_name)

        self.attack_name = os.path.join('{}'.format(args.attack_type))
        self.attack_path = os.path.join(args.base_path, 'attacker', self.attack_name)

        print(self.attack_path)

        # statistical attack
        prediction_scores = np.load(os.path.join(self.cls_path, 'prediction_scores.npy'), allow_pickle=True).item()

        train_entropy = entropy(prediction_scores['train']['preds'], base=2, axis=1)
        test_entropy = entropy(prediction_scores['test']['preds'], base=2, axis=1)
        acc, _ = classify_membership(train_entropy, test_entropy)
        print(acc)
