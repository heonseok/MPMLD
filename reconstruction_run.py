import os

dataset_list = [
    # 'adult'
    # 'location',
    # 'MNIST',
    # 'Fashion-MNIST',
    # 'CIFAR-10',
    'SVHN',
]

setsize_list = [
    # 100,
    # 500,
    # 1000,
    # 2000,
    10000,
    # 20000,
]

z_dim_list = [
    # '16',
    '64',
    # '128',
    # '256',
]

ref_ratio_list = [
    # 0.1,
    # 0.2,
    # 0.5,
    1.0,
]

lr_list = [
    0.001,
    # 0.01,
    # 0.1,
]

# recon, class_cz(+), class_mz(-), membership_cz(-), membership_mz(+)
weight_list = [
    # [100, 0, 1, 1, 0]
    # [100, 0, 1, 10, 0],

    # [100, 0, 10, 1, 0],
    # [100, 0, 10, 10, 0],
    # [10, 0, 1, 1, 0],
    # [10, 0, 10, 1, 0],
    # [10, 0, 1, 10, 0],
    # [10, 0, 10, 10, 0],

    # [10, 0, 0.1, 0.1, 0],
    # [10, 0, 0.1, 1, 0],
    # [100, 0, 0.1, 1, 0],
    # [50, 0, 0.1, 1, 0],
    # [50, 0, 1, 1, 0],
    # [1, 0, 0.01, 0.01, 0],
    # [1, 0, 0.1, 0.1, 0],
    # [1, 0, 0.01, 0.1, 0],

    # ref 1.0
    # [100, 0, 1, 1, 0],
    # [100, 0, 10, 1, 0],

    # ref 1.0 + permuted ref
    # [100, 0, 1, 1, 0], # best result at 0727 6:37
    # [100, 0, 10, 1, 0],
    # [100, 0, 1, 10, 0],
    # [100, 0, 10, 10, 0],
    # [100, 1, 1, 1, 1],

    # [100, 1, 0, 0, 0],
    # [100, 1, 0, 0, 0],
    # [100, 1, 0, 0, 0],
    # [100, 1, 0, 0, 0],

    # [1, 0, 1, 1, 0],
    # [1, 0, 10, 1, 0],
    # [1, 0, 1, 10, 0],
    # [1, 0, 10, 10, 0],
    # [1, 1, 1, 1, 1],

    # [1, 0, 1, 1, 0],
    # [1, 0, 1, 1, 0],
    [1, 0, 1, 2, 0],
    [1, 0, 1, 5, 0],
    [1, 0, 1, 10, 0],

    [1, 0, 0.5, 1, 0],
    [1, 0, 0.2, 1, 0],
    [1, 0, 0.1, 1, 0],

]

beta_list = [
    0.000001,
    # 0.00001,
    # 0.0001,
    # 0.001,
    # 0.01,
    # 0.1,
    # 1.0,
]

repeat_idx_list = [
    0,
    1,
    2,
    3,
    4,
]

setup_dict = {
    'train_reconstructor': '1',
    'reconstruct_datasets': '1',

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 0,
    'print_training': True,

    'output_dir': 'output0728',
}

for dataset in dataset_list:
    for beta in beta_list:
        for z_dim in z_dim_list:
            for setsize in setsize_list:
                for lr in lr_list:
                    for ref_ratio in ref_ratio_list:
                        for weight in weight_list:
                            for repeat_idx in repeat_idx_list:
                                args_list = []
                                target_setup_dict = setup_dict
                                target_setup_dict['dataset'] = dataset
                                target_setup_dict['z_dim'] = z_dim
                                target_setup_dict['setsize'] = setsize
                                target_setup_dict['repeat_idx'] = str(repeat_idx)
                                target_setup_dict['lr'] = str(lr)
                                target_setup_dict['ref_ratio'] = str(ref_ratio)
                                target_setup_dict['beta'] = str(beta)
                                target_setup_dict['recon_weight'] = str(weight[0])
                                target_setup_dict['class_cz_weight'] = str(weight[1])
                                target_setup_dict['class_mz_weight'] = str(weight[2])
                                target_setup_dict['membership_cz_weight'] = str(weight[3])
                                target_setup_dict['membership_mz_weight'] = str(weight[4])

                                args_list.append('python reconstruction_main.py')
                                for k, v in target_setup_dict.items():
                                    args_list.append('--{} {}'.format(k, v))

                                model = ' '.join(args_list)
                                print(model)
                                os.system(model)

