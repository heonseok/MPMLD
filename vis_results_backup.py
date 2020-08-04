import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

base_path = '/mnt/disk1/heonseok/MPMLD/output'


def plot_classification_result(dataset, clf_model, fig_path):
    clf_path = os.path.join(base_path, dataset, 'classifier', clf_model)
    df = pd.DataFrame()
    for repeat in range(5):
        try:
            clf_repeat_path = os.path.join(clf_path, 'repeat{}'.format(repeat))
            acc = np.load(os.path.join(clf_repeat_path, 'acc.npy'), allow_pickle=True).item()
            df = df.append(acc, ignore_index=True)
        except FileNotFoundError:
            continue
    df = df[['train', 'valid', 'test']]
    # print(clf_path)
    # print(df)
    sns.boxplot(data=df)
    plt.ylabel('Classification accuracy', fontdict={'size': 18})
    plt.title(clf_model, fontdict={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.5, 1.01)
    # plt.yticks(np.arange(0.1, 1.01, 0.1))
    # plt.axhline(0.1, ls='--', c='r')
    plt.tight_layout()
    if 'full_z' in clf_model or 'partial_z' in clf_model:
        clf_model = clf_model.replace('/', '_')
    plt.savefig(os.path.join(fig_path, '{}.jpg'.format(clf_model)))
    # plt.savefig(os.path.join(fig_path, 'classification.jpg'))
    plt.show()
    plt.close()


def plot_attack_result(dataset, clf_model, fig_path):
    df = pd.DataFrame()
    attack_type_list = ['stat', 'black']
    for repeat in range(5):
        for attack_type in attack_type_list:
            try:
                attack_path = os.path.join(base_path, dataset, 'attacker', clf_model, 'repeat{}'.format(repeat))
                acc = np.load(os.path.join(attack_path, attack_type, 'acc.npy'), allow_pickle=True)
                if attack_type == 'stat':
                    df = df.append({attack_type: acc}, ignore_index=True)
                else:
                    df = df.append({attack_type: acc.item()['test']}, ignore_index=True)
            except FileNotFoundError:
                continue

    sns.boxplot(data=df)
    plt.ylabel('Attack accuracy', fontdict={'size': 18})
    plt.title(clf_model, fontdict={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0., 1.01)
    plt.axhline(0.5, ls='--', c='r')
    plt.tight_layout()

    if 'full_z' in clf_model or 'partial_z' in clf_model:
        clf_model = clf_model.replace('/', '_')
    plt.savefig(os.path.join(fig_path, '{}.jpg'.format(clf_model)))
    plt.show()
    plt.close()


if __name__ == "__main__":

    if not os.path.exists('Figs'):
        os.mkdir('Figs')

    # target_models = dict()

    # clf_model_list = [
    #     # ('CIFAR-10', 'ResNet18_setsize1000_original'),
    #     # ('CIFAR-10', 'ResNet18_setsize1000_AE_z64_base/full_z'),
    #
    #     # ('CIFAR-10', 'ResNet18_setsize10000_original'),
    #     # ('CIFAR-10', 'ResNet18_setsize10000_AE_z64_base/full_z'),
    #
    #     ('CIFAR-10', 'ResNet50_setsize10000_original'),
    #     # ('CIFAR-10', 'ResNet50_setsize20000_original'),
    #     # ('CIFAR-10', 'ResNet50_setsize10000_AE_z64_base/partial_z'),
    #     # ('CIFAR-10', 'ResNet50_setsize10000_AE_z64_base/full_z'),
    #     # ('CIFAR-10', 'ResNet50_setsize10000_AE_z64_type1/partial_z'),
    #     # ('CIFAR-10', 'ResNet50_setsize10000_AE_z64_type1/full_z'),
    #     # ('CIFAR-10', 'ResNet50_setsize10000_AE_z64_type2/partial_z'),
    #     # ('CIFAR-10', 'ResNet50_setsize10000_AE_z64_type2/full_z'),
    #
    #     # ('CIFAR-10', 'ResNet101_setsize10000_original'),
    #     # ('CIFAR-10', 'ResNet101_setsize20000_original'),
    # ]

    # clf_model_list = [
    #     ('adult', 'FCN_setsize100_original'),
    #     # ('adult', 'FCN_setsize100_AE_z8_base/partial_z'),
    #     # ('adult', 'FCN_setsize100_AE_z8_type1/partial_z'),
    #     # ('adult', 'FCN_setsize100_AE_z8_type2/partial_z'),
    #     ('adult', 'FCN_setsize1000_original'),
    #     ('adult', 'FCN_setsize10000_original')
    # ]

    # clf_model_list = [
    #     ('MNIST', 'ConvClassifier_setsize200_original'),
    #     ('MNIST', 'ConvClassifier_setsize300_original'),
    #     ('MNIST', 'ConvClassifier_setsize400_original'),
    #     ('MNIST', 'ConvClassifier_setsize500_original'),
    #     ('MNIST', 'ConvClassifier_setsize1000_original'),
    #     ('MNIST', 'ConvClassifier_setsize10000_original'),
    # ]

    # clf_model_list = [
    #     # ('Fashion-MNIST', 'ConvClassifier_setsize200_original'),
    #     # ('Fashion-MNIST', 'ConvClassifier_setsize300_original'),
    #     # ('Fashion-MNIST', 'ConvClassifier_setsize400_original'),
    #     ('Fashion-MNIST', 'ConvClassifier_setsize500_original'),
    #     ('Fashion-MNIST', 'ConvClassifier_setsize1000_original'),
    #     ('Fashion-MNIST', 'ConvClassifier_setsize10000_original'),
    # ]

    clf_model_list = [
        # ('location', 'FCNClassifier_setsize500_original'),
        # ('location', 'FCNClassifier_setsize1000_original'),
        # ('location', 'FCNClassifier_setsize2000_original'),
        ('location', 'FCNClassifier_setsize2000_AE_z64_base/full_z'),
        ('location', 'FCNClassifier_setsize2000_AE_z64_type1/full_z'),
        ('location', 'FCNClassifier_setsize2000_AE_z64_type1/partial_z'),
        ('location', 'FCNClassifier_setsize2000_AE_z64_type2/full_z'),
        ('location', 'FCNClassifier_setsize2000_AE_z64_type2/partial_z'),
    ]

    for (dataset, clf_model) in clf_model_list:
        fig_path = os.path.join('Figs', dataset, 'classification')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plot_classification_result(dataset, clf_model, fig_path)

        fig_path = os.path.join('Figs', dataset, 'attack')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plot_attack_result(dataset, clf_model, fig_path)
