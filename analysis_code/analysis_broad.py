#  %%
from analysis_utils import *

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

# recon_df, class_df, attack_df = collate_dfs(base_path, dataset_list, description_list, beta_list, z_dim_list, setsize_list, recon_lr_list, ref_ratio_list, weight_list, small_recon_weight_list, recon_train_batch_size, early_stop_flag, target_epoch)
recon_df, class_df, attack_df = collate_dfs(base_path, dataset_list, description_list, beta_list, z_dim_list, setsize_list, recon_lr_list, ref_ratio_list, weight_list, small_recon_weight_list, recon_train_batch_size, recon_type_list, early_stop_flag, target_epoch)

# %%
## BETA
expr = 'small_recon_weight == 0.01'
column_type = 'beta'

draw_results(recon_df, class_df, attack_df, expr, column_type)

# %%
