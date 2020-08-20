import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

base_path = os.path.join('/mnt/disk1/heonseok/MPMLD')
if not os.path.exists('Figs'):
    os.mkdir('Figs')

REPEAT = 5

# -------------------------------------------------------------------------------------------------------------------- #
def main():
    dataset = 'SVHN'
    description = 'baseline'
    bs_list = [4, 8, 16, 32, 64, 128, 256]
    model_prefix = 'ResNet18_lr0.0001_bs'

    class_df = pd.DataFrame()
    attack_df = pd.DataFrame()

    for bs in bs_list:
        class_model_path = os.path.join(base_path, dataset, description, 'raw_setsize5000/classification', model_prefix + str(bs))
        attack_model_path = os.path.join(base_path, dataset, description, 'raw_setsize5000/attack', model_prefix + str(bs), 'black')

        for repeat_idx in range(REPEAT):
            class_repeat_path = os.path.join(class_model_path, 'repeat' + str(repeat_idx), 'acc.npy')
            attack_repeat_path = os.path.join(attack_model_path, 'repeat' + str(repeat_idx), 'acc.npy')
            class_acc_dict = np.load(class_repeat_path, allow_pickle=True).item()
            attack_acc_dict = np.load(attack_repeat_path, allow_pickle=True).item()

            for dataset_type in ['train', 'valid', 'test']:
                class_df = class_df.append({'bs': bs, 'dataset': dataset_type, 'acc': class_acc_dict[dataset_type]},
                                           ignore_index=True)

            attack_df = attack_df.append({'bs': bs, 'attack_type': 'black', 'acc': attack_acc_dict['test']},
                                         ignore_index=True)

    sns.barplot(x='bs', y='acc', hue='dataset', data=class_df)
    plt.ylim(0.5, 1.01)
    plt.tight_layout()
    img_dir = os.path.join('Figs', dataset, description, 'classification_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format('raw_setsize5000_bs'))
    plt.savefig(img_path)
    plt.close()
    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    sns.barplot(x='bs', y='acc', data=attack_df)
    plt.ylim(0.5, 0.9)
    plt.tight_layout()
    img_dir = os.path.join('Figs', dataset, description, 'attack_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format('raw_setsize5000_bs'))
    plt.savefig(img_path)
    plt.close()
    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    print('Finish!')


if __name__ == '__main__':
    main()
