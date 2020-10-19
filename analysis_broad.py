#  %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
def collate_recon_results(recon_path, recon_dict):
    df = pd.DataFrame()
    for repeat_idx in range(5):
        repeat_path = os.path.join(recon_path, 'repeat{}'.format(repeat_idx))

        if not early_stop_flag:
            class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc{:03d}.npy'.format(target_epoch)), allow_pickle=True).item()
            membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc{:03d}.npy'.format(target_epoch)), allow_pickle=True).item()
        else:
            class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc.npy'), allow_pickle=True).item()
            membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc.npy'), allow_pickle=True).item()

        for z_type in ['pn', 'pp', 'np', 'nn']:
            acc_dict = {'disc_type': 'class', 'z_type': z_type, 'acc': class_acc_dict[z_type]}
            df = df.append({**recon_dict, **acc_dict}, ignore_index=True)
            acc_dict = {'disc_type': 'membership', 'z_type': z_type, 'acc': membership_acc_dict[z_type]}
            df = df.append({**recon_dict, **acc_dict}, ignore_index=True)
    
    return df 

def collate_class_results(class_path, class_dict, recon_type):
    df = pd.DataFrame()
    for repeat_idx in range(5):

        if 'raw' in class_path:
            repeat_path = os.path.join(class_path, 'repeat' + str(repeat_idx), 'acc.npy')
            recon_type = 'raw'
        else:
            repeat_path = os.path.join(class_path, recon_type, 'repeat' + str(repeat_idx), 'acc.npy')

        result_dict = np.load(repeat_path, allow_pickle=True).item()
        for dataset_type in ['train', 'valid', 'test']:
            acc_dict = {'recon_type': recon_type, 'dataset': dataset_type, 'acc': result_dict[dataset_type]}
            df = df.append({**class_dict, **acc_dict}, ignore_index=True)


    return df



def collate_attack_results(class_path, class_dict):
    df = pd.DataFrame()
    for repeat_idx in range(5):
        pass

# %%
dataset_list = [
    # 'adult'
    # 'location',
    # 'MNIST',
    # 'Fashion-MNIST',
    'SVHN',
    # 'CIFAR-10',
]


setsize_list = [
    # 50,
    # 100,
    # 200,
    # 300,
    # 400,
    # 500,
    # 1000,
    # 2000,
    # 4000,
    5000,
    # 10000,
    # 20000,
]

z_dim_list = [
    # '16',
    # '32',
    '64',
    # '128',
    # '256',
]

ref_ratio_list = [
    # 0.1,
    # 0.2,
    # 0.5,
    1.0,
    # 2.0,
]

recon_lr_list = [
    # 0.0001,
    0.001,
    # 0.01,
    # 0.1,
]

# recon, real_fake, class_pos, class_neg, membership_pos, membership_neg
weight_list = [
    # [1., 0., 0., 0., 0., 0.],
    # [1., 1., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1., 1.],
    # [1., 1., 1., 1., 1., 1.],
    [1., 0., 1., 2., 1., 2.],
]

beta_list = [
    # 0.0
    # 0.000001,
    # 0.00001,
    # 0.0001,
    # 0.001,
    0.01,
    0.1,
    # 1.0,
]

small_recon_weight_list = [
    # 0.,
    # 0.001,
    0.01,
    0.1,
    # 1.,
]

recon_type_list = [
    # 'raw',
    'pn_pp_np_nn',  # [1, 1, 1, 1]
    'pn_pp_nn',  # [1, 1, 0, 1]
    'pn_pp',  # [1, 1, 0, 0]
    'pp_np',  # [0, 1, 1, 0]
    'np_nn',  # [0, 0, 1, 1]
    # 'pn',  # [1, 0, 0, 0]
    # 'pp',  # [1, 0, 0, 0]
    # 'np',  # [1, 0, 0, 0]
    # 'nn',  # [1, 0, 0, 0]
]

base_path = '/mnt/disk1/heonseok/MPMLD'
description_list = [
    # '1006debug',
    # '1008',
    '1012normalized_tanh',
]


recon_train_batch_size = 64

early_stop_flag = False
target_epoch = 200


# %%
recon_df = pd.DataFrame()
class_df = pd.DataFrame()

for dataset in dataset_list:
    for description in description_list:
        for beta in beta_list:
            for z_dim in z_dim_list:
                for setsize in setsize_list:
                    for recon_lr in recon_lr_list:
                        for ref_ratio in ref_ratio_list:
                            for weight in weight_list:
                                for small_recon_weight in small_recon_weight_list:

                                    reconstruction_name = '{}{}_{}Enc_{}Disc_z{}_setsize{}_lr{}_bs{}_ref{}_rw{}_rf{}_cp{}_cn{}_mp{}_mn{}_sr{}'.format(
                                        'VAE',
                                        beta,
                                        'distinct',
                                        'distinct',
                                        z_dim,
                                        setsize, 
                                        recon_lr,
                                        recon_train_batch_size,
                                        ref_ratio,
                                        weight[0],
                                        weight[1],
                                        weight[2],
                                        weight[3],
                                        weight[4],
                                        weight[5],
                                        small_recon_weight,
                                    )

                                    weight_str = '[' + ', '.join([str(elem) for elem in weight]) + ']'

                                    recon_dict = {
                                        'dataset': dataset,
                                        'description': description,
                                        'beta': float(beta), 
                                        'z_dim' : int(z_dim),
                                        'setsize' : int(setsize),
                                        'recon_lr': float(recon_lr),
                                        'recon_train_batch_size': int(recon_train_batch_size),
                                        'ref_ratio': float(ref_ratio),
                                        #    'recon_weight': weight[0],
                                        #    'realfake_weight': weight[1],
                                        #    'class_pos_weight': weight[2],
                                        #    'class_neg_weight': weight[3],
                                        #    'membership_pos_weight': weight[4],
                                        #    'membership_neg_weight': weight[5],
                                        'weights' : weight_str, 
                                        'small_recon_weight': float(small_recon_weight),
                                    }

                                    try:
                                        recon_path = os.path.join(base_path, dataset, description, reconstruction_name, 'reconstruction')
                                        # print(recon_path)
                                        recon_df = pd.concat([recon_df, collate_recon_results(recon_path, recon_dict)])

                                    except FileNotFoundError:
                                        pass

                                    for recon_type in recon_type_list:
                                        try:
                                            # print(recon_path)
                                            # class_path = os.path.join(base_path, dataset, description)
                                            class_path = os.path.join(base_path, dataset, description, reconstruction_name, 'classification', 'ResNet18_lr0.0001_bs32')
                                            # print(class_path)
                                            class_df = pd.concat([class_df, collate_class_results(class_path, recon_dict, recon_type)])

                                        except FileNotFoundError:
                                            # print('There is no file : {}'.format(class_path))
                                            pass



# %%
# small recon weight 
# target_df = recon_df 
# target_df = recon_df.query('small_recon_weight == 0')
# print(weight_str)
# target_df = recon_df.query('weights == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]')
# print(target_df)
# sns.catplot(data=target_df, x='small_recon_weight', y='acc', hue='z_type', col='disc_type', kind='box')
# sns.catplot(data=target_df, x='disc_type', y='acc', hue='z_type', col='small_recon_weight', kind='box')

# target_df = recon_df.query('small_recon_weight == 0')
# sns.catplot(data=target_df, x='small_recon_weight', y='acc', hue='z_type', col='disc_type', kind='box')
# sns.catplot(data=target_df, x='disc_type', y='acc', hue='z_type', col='small_recon_weight', kind='box')

# %%
## WEIGHTS
target_recon_df = recon_df.copy()
target_recon_df = target_recon_df.query('z_dim == 64')
print('z_dim : 64')
sns.catplot(data=target_recon_df, x='disc_type', y='acc', hue='z_type', col='weights', kind='box')

target_recon_df = recon_df.copy()
target_recon_df = target_recon_df.query('z_dim == 32')
print('z_dim : 32')
sns.catplot(data=target_recon_df, x='disc_type', y='acc', hue='z_type', col='weights', kind='box')

# %%
## BETA
target_recon_df = recon_df.copy()
target_recon_df = target_recon_df.query('small_recon_weight == 0.01')
sns.catplot(data=target_recon_df, x='disc_type', y='acc', hue='z_type', col='beta', kind='box')

target_class_df = class_df.copy()
target_class_df = target_class_df.query('small_recon_weight == 0.01')
print(target_class_df)
sns.catplot(data=target_class_df, x='recon_type', y='acc', hue='dataset', col='beta', kind='box')


# %%
## Description 

