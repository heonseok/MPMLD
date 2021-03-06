# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_path = os.path.join('/mnt/disk1/heonseok/MPMLD')
if not os.path.exists('Figs'):
    os.mkdir('Figs')

REPEAT = range(1,5) 
unit_length = 10

eartly_stop_flag = False
target_epoch = 200

# %%


def collate_reconstructions(dataset, description, model, recon_type_list):
    model_path = os.path.join( base_path, dataset, description, model, 'reconstruction')

    fig, axes = plt.subplots(nrows=REPEAT, ncols=len(recon_type_list), figsize=( unit_length*len(recon_type_list), unit_length*REPEAT))
    for repeat_idx in REPEAT:
        for recon_idx, recon_type in enumerate(recon_type_list):
            ax = axes[repeat_idx][recon_idx]
            if eartly_stop_flag:
                ax.imshow(mpimg.imread(os.path.join(model_path, 'repeat' + str(repeat_idx), recon_type + '.png')))
            else:
                ax.imshow(mpimg.imread(os.path.join(model_path, 'repeat' + str(repeat_idx), recon_type + '{}.png'.format(target_epoch))))

            if repeat_idx == 0:
                # print(recon_type)
                # ax.title.set_text(recon_type, fontsize=10)
                ax.set_title(recon_type, fontsize=80)

            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()

    plt.tight_layout()
    img_dir = os.path.join('Figs', dataset, description, 'recon_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.jpeg'.format(model))
    plt.savefig(img_path)
    plt.show()
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))


def collate_disentanglement_result(dataset, description, model):
    model_path = os.path.join( base_path, dataset, description, model, 'reconstruction')
    df = pd.DataFrame()
    for repeat_idx in REPEAT:
        repeat_path = os.path.join(model_path, 'repeat{}'.format(repeat_idx))
        if not early_stop_flag:
            class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc{:03d}.npy'.format(target_epoch)), allow_pickle=True).item()
            membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc{:03d}.npy'.format(target_epoch)), allow_pickle=True).item()
        else:
            class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc.npy'), allow_pickle=True).item()
            membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc.npy'), allow_pickle=True).item()
        for z_type in ['pn', 'pp', 'np', 'nn']:
            df = df.append({'description': description, 'disc_type': 'class', 'z_type': z_type, 'acc': class_acc_dict[z_type]}, ignore_index=True)
            df = df.append({'description': description, 'disc_type': 'membership', 'z_type': z_type, 'acc': membership_acc_dict[z_type]}, ignore_index=True)

    sns.boxplot(x='disc_type', y='acc', hue='z_type', data=df)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()

    img_dir = os.path.join('Figs', dataset, description, 'disentanglement_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.show()
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    return df


def collate_classification_result(dataset, description, model, recon_type_list):
    model_path = os.path.join(base_path, dataset, description, model, 'classification', 'ResNet18_lr0.0001_bs32')
    df = pd.DataFrame()

    for recon_idx, recon_type in enumerate(recon_type_list):
        for repeat_idx in REPEAT:
            if 'raw' in model:
                repeat_path = os.path.join( model_path, 'repeat' + str(repeat_idx), 'acc.npy')
                recon_type = 'raw'
            else:
                repeat_path = os.path.join( model_path, recon_type, 'repeat' + str(repeat_idx), 'acc.npy')

            acc_dict = np.load(repeat_path, allow_pickle=True).item()
            for dataset_type in ['train', 'valid', 'test']:
                df = df.append({'description': description, 'recon': recon_type, 'dataset': dataset_type, 'acc': acc_dict[dataset_type]}, ignore_index=True)
        if 'raw' in model:
            break

    sns.boxplot(x='recon', y='acc', hue='dataset', data=df)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()

    img_dir = os.path.join('Figs', dataset, description, 'classification_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.show()
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    return df


def collate_attack_result(dataset, description, model, recon_type_list):
    model_path = os.path.join( base_path, dataset, description, model, 'attack', 'ResNet18_lr0.0001_bs32')

    attack_type_list = [
        'black',
        'white',
    ]

    df = pd.DataFrame()
    for recon_idx, recon_type in enumerate(recon_type_list):
        for attack_type in attack_type_list:
            for repeat_idx in REPEAT:
                if 'raw' in model:
                    repeat_path = os.path.join(model_path, attack_type, 'repeat' + str(repeat_idx), 'acc.npy')
                    recon_type = 'raw'
                else:
                    repeat_path = os.path.join(model_path, recon_type, attack_type, 'repeat' + str(repeat_idx), 'acc.npy')
                acc_dict = np.load(repeat_path, allow_pickle=True).item()
                df = df.append({'description': description, 'recon': recon_type, 'attack_type': attack_type, 'acc': acc_dict['test']}, ignore_index=True)

    sns.boxplot(x='recon', y='acc', hue='attack_type', data=df)
    plt.ylim(0.4, 0.8)
    plt.tight_layout()

    img_dir = os.path.join('Figs', dataset, description, 'attack_collated')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, '{}.png'.format(model))
    plt.savefig(img_path)
    plt.show()
    plt.close()

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

    return df


# %%
dataset = 'SVHN'
# dataset = 'MNIST'

# description = '0825_4typesDisentanglement_small_recon'
# description = '0915'
# description = '0924'
# description = '1012normalized_tanh'
# description = '1013non_iid_color_zero'
description = '1016_cosine_opt'
print(description)

early_stop_flag = False
target_epoch = 200

model_list = [
    # 'VAE0.0_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp0.0_cn0.0_mp0.0_mn0.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp0.0_cn0.0_mp0.0_mn0.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.0',

    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp0.5_cn0.5_mp1.0_mn1.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp2.0_mn2.0_sr0.0',

    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.001',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs32_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',
    # 'raw_setsize5000',

    # 1002
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',

    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp1.0_mn2.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp2.0_mn1.0_sr0.0',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp2.0_mn2.0_sr0.0',

    # 'VAE0.01_distinctEnc_distinctDisc_z64_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',
    # 'VAE0.01_distinctEnc_distinctDisc_z64_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',

    # 'VAE0.01_distinctEnc_distinctDisc_z32_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',
    # 'VAE0.01_distinctEnc_distinctDisc_z32_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',
    # 'VAE0.01_distinctEnc_distinctDisc_z128_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf1.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',

    'VAE0.01_distinctEnc_distinctDisc_z64_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn1.0_mp1.0_mn1.0_sr0.01',
    # 'VAE0.01_distinctEnc_distinctDisc_z64_setsize5000_lr0.001_bs64_ref1.0_rw1.0_rf0.0_cp1.0_cn2.0_mp1.0_mn2.0_sr0.01',
]


recon_type_list = [
    'pn_pp_np_nn',  # [1, 1, 1, 1]
    'pn_pp_nn',  # [1, 1, 0, 1]
    'pn_pp',  # [1, 1, 0, 0]
    'pn_nn',  # [1, 0, 0, 1]
    'pp_np',  # [0, 1, 1, 0]
    'np_nn',  # [0, 0, 1, 1]
    # 'pn',  # [1, 0, 0, 0]
    # 'pp',  # [1, 0, 0, 0]
    # 'np',  # [1, 0, 0, 0]
    # 'nn',  # [1, 0, 0, 0]
]


for model in model_list:
    print(model)
    if 'raw' not in model:
        # collate_reconstructions(dataset, description, model, recon_type_list)
        recon_df = collate_disentanglement_result(dataset, description, model)
        pass
    class_df = collate_classification_result(dataset, description, model, recon_type_list)
    attack_df = collate_attack_result(dataset, description, model, recon_type_list)

print('Finish!')


# %%
