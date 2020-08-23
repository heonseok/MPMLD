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
    # description = 'baseline'
    # bs_list = [8]
    # class_model = 'ResNet18_lr0.0001_bs8'

    data_list = [
        ('baseline', 'raw_setsize5000', 'ResNet18_lr0.0001_bs8', 'raw'),
        # ('VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', 'VAE(2)'),

        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', '22b'),
        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cr_mr', '22r'),

        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', '28b'),
        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cr_mr', '28r'),

        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs4_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', '48b'),
        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs4_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cr_mr', '48r'),

        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs8_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', '88b'),
        # ('baseline', 'VAE0.1_z64_setsize5000_lr0.001_bs8_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cr_mr', '88r'),

        ('0818bs2', 'VAE0.1_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0', 'ResNet18/cb_mb', '22b+AG'),
        ('0821noDE', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs2/cb_mb', '22b'),
        # ('0821noDE', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs4/cb_mb', '24b'),
        # ('0821noDE', 'VAE0.1_z64_setsize5000_lr0.001_bs2_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', '28b'),
        # ('0818bs2', 'VAE0.1_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0', 'ResNet18/cr_mr', 'cm22r'),
        # ('VAE0.1_z64_setsize5000_lr0.001_bs8_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cr_mr', 'VAE(8r)'),
        # ('VAE0.1_z64_setsize5000_lr0.001_bs8_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', 'VAE(cc)'),
        # ('VAE0.1_z64_setsize5000_lr0.001_bs8_ref1.0_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0', 'ResNet18_lr0.0001_bs8/cb_mb', 'VAE(cmmc)'),
    ]

    class_df = pd.DataFrame()
    attack_df = pd.DataFrame()

    for description, data, class_model, data_name in data_list:
        class_model_path = os.path.join(base_path, dataset, description, data, 'classification', class_model)
        attack_model_path = os.path.join(base_path, dataset, description, data, 'attack', class_model, 'black')

        for repeat_idx in range(REPEAT):
            class_repeat_path = os.path.join(class_model_path, 'repeat' + str(repeat_idx), 'acc.npy')
            attack_repeat_path = os.path.join(attack_model_path, 'repeat' + str(repeat_idx), 'acc.npy')
            class_acc_dict = np.load(class_repeat_path, allow_pickle=True).item()
            attack_acc_dict = np.load(attack_repeat_path, allow_pickle=True).item()

            for dataset_type in ['train', 'valid', 'test']:
                class_df = class_df.append({'data': data_name, 'dataset': dataset_type, 'acc': class_acc_dict[dataset_type]},
                                           ignore_index=True)

            attack_df = attack_df.append({'data': data_name, 'attack_type': 'black', 'acc': attack_acc_dict['test']},
                                         ignore_index=True)

    # sns.barplot(x='data', y='acc', hue='dataset', data=class_df)
    sns.boxplot(x='data', y='acc', hue='dataset', data=class_df)
    plt.ylim(0.7, 1.01)
    plt.tight_layout()
    img_dir = os.path.join('Figs', dataset, description, 'classification_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format('recon'))
    plt.savefig(img_path)
    plt.close()
    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    # sns.barplot(x='data', y='acc', data=attack_df)
    sns.boxplot(x='data', y='acc', data=attack_df)
    plt.ylim(0.49, 0.65)
    plt.tight_layout()
    img_dir = os.path.join('Figs', dataset, description, 'attack_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format('recon'))
    plt.savefig(img_path)
    plt.close()
    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    print('Finish!')


if __name__ == '__main__':
    print('Recon')
    main()
