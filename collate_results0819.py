import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

base_path = os.path.join('/mnt/disk1/heonseok/MPMLD')
if not os.path.exists('Figs'):
    os.mkdir('Figs')

REPEAT = 5


# -------------------------------------------------------------------------------------------------------------------- #
def collate_reconstructions(dataset, description, model, recon_type_list):
    model_path = os.path.join(base_path, dataset, description, model, 'reconstruction')

    plt.figure(1, figsize=(9, 6))
    for recon_idx, recon_type in enumerate(recon_type_list):
        for repeat_idx in range(REPEAT):
            plt.subplot(len(recon_type_list), REPEAT, repeat_idx + recon_idx * REPEAT + 1)
            plt.imshow(mpimg.imread(os.path.join(model_path, 'repeat' + str(repeat_idx), recon_type + '.png')))
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout(pad=0.1)
    img_dir = os.path.join('Figs', dataset, description, 'recon_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))


def collate_disentanglement_result(dataset, description, model):
    model_path = os.path.join(base_path, dataset, description, model, 'reconstruction')
    df = pd.DataFrame()
    for repeat_idx in range(REPEAT):
        repeat_path = os.path.join(model_path, 'repeat{}'.format(repeat_idx), 'acc.npy')
        acc_dict = np.load(repeat_path, allow_pickle=True).item()
        for disc_type in ['class', 'membership']:
            for z_type in ['fz', 'cz', 'mz']:
                df = df.append({'disc_type': disc_type, 'z_type': z_type, 'acc': acc_dict[disc_type + '_' + z_type]},
                               ignore_index=True)

    sns.boxplot(x='disc_type', y='acc', hue='z_type', data=df)
    plt.ylim(0., 1.)
    plt.tight_layout()

    img_dir = os.path.join('Figs', dataset, description, 'disentanglement_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))


def collate_classification_result(dataset, description, model):
    model_path = os.path.join(base_path, dataset, description, model, 'classification', model)
    df = pd.DataFrame()
    for repeat_idx in range(REPEAT):
        repeat_path = os.path.join(model_path, 'repeat' + str(repeat_idx), 'acc.npy')
        acc_dict = np.load(repeat_path, allow_pickle=True).item()
        for dataset_type in ['train', 'valid', 'test']:
            df = df.append({'dataset': dataset_type, 'acc': acc_dict[dataset_type]}, ignore_index=True)

    sns.boxplot(x='recon', y='acc', hue='dataset', data=df)
    plt.ylim(0., 1.01)
    plt.tight_layout()

    img_dir = os.path.join('Figs', dataset, description, 'classification_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))


def collate_attack_result(dataset, description, model, recon_type_list):
    model_path = os.path.join(base_path, dataset, description, model, 'attack', 'ResNet18')
    attack_type_list = [
        'black',
        # 'white',
    ]

    df = pd.DataFrame()
    for recon_idx, recon_type in enumerate(recon_type_list):
        for attack_type in attack_type_list:
            for repeat_idx in range(REPEAT):
                repeat_path = os.path.join(model_path, recon_type, attack_type, 'repeat' + str(repeat_idx), 'acc.npy')
                acc_dict = np.load(repeat_path, allow_pickle=True).item()
                df = df.append({'recon': recon_type, 'attack_type': attack_type, 'acc': acc_dict['test']},
                               ignore_index=True)

    sns.boxplot(x='recon', y='acc', hue='attack_type', data=df)
    plt.ylim(0.4, 0.8)
    plt.tight_layout()

    img_dir = os.path.join('Figs', dataset, description, 'attack_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))


# -------------------------------------------------------------------------------------------------------------------- #
def main():
    dataset = 'SVHN'
    description = 'raw_setsize5000'
    bs_list = [8, 16, 32, 64, 128, 256]
    model_prefix = 'ResNet18_lr0.0001_bs'

    class_df = pd.DataFrame()
    attack_df = pd.DataFrame()

    for bs in bs_list:
        class_model_path = os.path.join(base_path, dataset, description, 'classification', model_prefix + str(bs))
        attack_model_path = os.path.join(base_path, dataset, description, 'attack', model_prefix + str(bs), 'black')

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
    plt.ylim(0., 1.01)
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
