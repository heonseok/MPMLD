import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import numpy as np
import pandas as pd
import seaborn as sns

base_path = os.path.join('/mnt/disk1/heonseok/MPMLD')
if not os.path.exists('Figs'):
    os.mkdir('Figs')


def collate_reconstructions(dataset, description, model):
    model_path = os.path.join(base_path, dataset, description, model, 'reconstruction')
    recon_type_list = [
        'cb_mb.png',
        'cz_mb.png',
        'cb_mz.png',
    ]

    plt.figure(1, figsize=(9, 6))
    for recon_idx, recon_type in enumerate(recon_type_list):
        for repeat_idx in range(5):
            plt.subplot(3, 5, repeat_idx + recon_idx * 5 + 1)
            plt.imshow(mpimg.imread(os.path.join(model_path, 'repeat{}'.format(repeat_idx), recon_type)))
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
    for repeat_idx in range(5):
        repeat_path = os.path.join(model_path, 'repeat{}'.format(repeat_idx), 'acc.npy')
        acc_dict = np.load(repeat_path, allow_pickle=True).item()
        df = df.append(acc_dict, ignore_index=True)
    df = df[['class_fz', 'class_cz', 'class_mz', 'membership_fz', 'membership_cz', 'membership_mz']]
    df = df.rename(columns={'membership_fz': 'mem_fz', 'membership_cz': 'mem_cz', 'membership_mz': 'mem_mz'})
    sns.boxplot(data=df)
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


# -------------------------------------------------------------------------------------------------------------------- #
def main():
    dataset = 'SVHN'
    description = '0803weights'
    model_list = [
        'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0',

        'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0',
        'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc0.0_mm0.0',
        'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc1.0_mm0.0',
        'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm1.0',

        'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0',
    ]

    for model in model_list:
        print(model)
        # collate_reconstructions(dataset, description, model)
        collate_disentanglement_result(dataset, description, model)


if __name__ == '__main__':
    main()
