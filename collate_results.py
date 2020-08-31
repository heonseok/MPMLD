import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
        repeat_path = os.path.join(model_path, 'repeat{}'.format(repeat_idx))
        class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc.npy'), allow_pickle=True).item()
        membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc.npy'), allow_pickle=True).item()
        for z_type in ['pn', 'pp', 'np', 'nn']:
            df = df.append({'disc_type': 'class', 'z_type': z_type, 'acc': class_acc_dict[z_type]},
                           ignore_index=True)
            df = df.append({'disc_type': 'membership', 'z_type': z_type, 'acc': membership_acc_dict[z_type]},
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


def collate_classification_result(dataset, description, model, recon_type_list):
    model_path = os.path.join(base_path, dataset, description, model, 'classification', 'ResNet18_lr0.0001_bs32')
    df = pd.DataFrame()
    for recon_idx, recon_type in enumerate(recon_type_list):
        for repeat_idx in range(REPEAT):
            repeat_path = os.path.join(model_path, recon_type, 'repeat' + str(repeat_idx), 'acc.npy')
            acc_dict = np.load(repeat_path, allow_pickle=True).item()
            for dataset_type in ['train', 'valid', 'test']:
                df = df.append({'recon': recon_type, 'dataset': dataset_type, 'acc': acc_dict[dataset_type]},
                               ignore_index=True)

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
    model_path = os.path.join(base_path, dataset, description, model, 'attack', 'ResNet18_lr0.0001_bs32')
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
    # description = '0803weights'
    # model_list = [
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0',
    #     #
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc0.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc1.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm1.0',
    #     #
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc0.0_mm1.0',
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc1.0_cm0.0_mc1.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc1.0_cm0.0_mc0.0_mm1.0',
    #
    #     # 'VAE1e-06_z64_setsize10000_lr0.001_ref0.1_rw1.0_cc1.0_cm1.0_mc1.0_mm1.0',
    # ]

    # description = '0804set'
    # model_list = [
    #     'VAE1e-06_z64_setsize1000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0',
    #     'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0',
    #     'VAE1e-06_z64_setsize4000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0',
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm0.0',
    # ]

    # description = '0804style'
    # model_list = [
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc0.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc1.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc0.0_cm0.0_mc0.0_mm1.0',
    #
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0',
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref0.1_rw1.0_cc0.0_cm1.0_mc0.0_mm1.0',
    # ]

    # description = '0805ref1.0style'
    # model_list = [
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm1.0',
    # ]

    # description = '0805class_conditional_mem_disc'
    # model_list = [
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm1.0_mc0.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm0.0_mc1.0_mm0.0',
    #     # 'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm0.0_mc0.0_mm1.0',
    #
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc1.0_cm0.0_mc0.0_mm1.0',
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0',
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc1.0_cm0.0_mc1.0_mm0.0',
    #     'VAE1e-06_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm1.0_mc0.0_mm1.0',
    # ]

    # description = '0818bs2'
    # model_list = [
    #     'VAE0.1_z64_setsize5000_lr0.001_ref1.0_rw1.0_cc0.0_cm1.0_mc1.0_mm0.0',
    # ]

    description = '0825_4typesDisentanglement_small_recon'
    model_list = [
        'VAE0.01_distinctEnc_sharedDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_cp1.0_cn1.0_mp1.0_mn1.0',
        'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_cp1.0_cn1.0_mp1.0_mn1.0',
    ]

    recon_type_list = [
        'pn_pp_np_nn',  # [1, 1, 1, 1]
        'pn_pp_nn',  # [1, 1, 0, 1]
        'pn_pp',  # [1, 1, 0, 0]
        'pp_np',  # [0, 1, 1, 0]
        'np_nn',  # [0, 0, 1, 1]
        'pn',  # [1, 0, 0, 0]
        'pp',  # [1, 0, 0, 0]
        'np',  # [1, 0, 0, 0]
        'nn',  # [1, 0, 0, 0]

        # 'cb_mb',
        # 'cb_mz',
        # 'cz_mb',

        # 'cb_mb_sb',
        # 'cb_mb_sz',
        # 'cb_mz_sb',
        # 'cb_mz_sz',
        # 'cz_mb_sb',
        # 'cz_mb_sz',
    ]
    for model in model_list:
        print(model)
        collate_reconstructions(dataset, description, model, recon_type_list)
        collate_disentanglement_result(dataset, description, model)
        collate_classification_result(dataset, description, model, recon_type_list)
        collate_attack_result(dataset, description, model, recon_type_list)

    print('Finish!')


if __name__ == '__main__':
    main()
