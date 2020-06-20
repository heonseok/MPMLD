import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

base_path = '/mnt/disk1/heonseok/MPMLD'


def plot_classification_result(clf_model):
    clf_path = os.path.join(base_path, 'classifier', clf_model)
    df = pd.DataFrame()
    for repeat in range(5):
        clf_repeat_path = os.path.join(clf_path, 'repeat{}'.format(repeat))
        acc = np.load(os.path.join(clf_repeat_path, 'acc.npy'), allow_pickle=True).item()
        df = df.append(acc, ignore_index=True)
    df = df[['train', 'valid', 'test']]
    sns.boxplot(data=df)
    plt.ylabel('Classification accuracy', fontdict={'size': 18})
    plt.title(clf_model, fontdict={'size': 20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.axhline(50, ls='--', c='r')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_attack_result(clf_model):
    df = pd.DataFrame()
    attack_type_list = ['stat', 'black']
    for repeat in range(5):
        for attack_type in attack_type_list:
            attack_path = os.path.join(base_path, 'attacker', clf_model, 'repeat{}'.format(repeat))
            acc = np.load(os.path.join(attack_path, attack_type, 'acc.npy'), allow_pickle=True)
            if attack_type == 'stat':
                df = df.append({attack_type:acc}, ignore_index=True)
            else:
                df = df.append({attack_type:acc.item()['test']}, ignore_index=True)

    sns.boxplot(data=df)
    plt.ylabel('Attack accuracy', fontdict={'size': 18})
    plt.title(clf_model, fontdict={'size': 20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.axhline(50, ls='--', c='r')
    plt.tight_layout()
    plt.show()
    plt.close()


def vis_classification_result():
    clf_model_list = [
        'ResNet18_setsize1000',
    ]

    for clf_model in clf_model_list:
        plot_classification_result(clf_model)
        plot_attack_result(clf_model)


if __name__ == "__main__":
    vis_classification_result()
    # vis_attack_result()
