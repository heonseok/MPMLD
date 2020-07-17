import os
import sys
import datetime

dataset_list = [
    # 'adult'
    'location',
    # 'MNIST',
    # 'Fashion-MNIST',
    # 'CIFAR-10',
]

setsize_list = [
    # 100,
    # 1000,
    # 500,
    # 1000,
    2000,
    # 10000,
    # 20000,
]

reconstruction_model_list = [
    # 'AE',
    'VAE'
]

z_dim_list = [
    # '16',
    '64',
]

disentanglement_type_list = [
    # 'base',
    # 'type1',
    # 'type2',
    # 'type3',
    # 'type4',
    'type5',
]

ref_ratio_list = [
    0.1,
    # 0.2,
    # 0.5,
    # 1.0,
]

architecture_list = [
    # 'A',
    # 'B',
    # 'C',
    'D',
]

lr_list = [
    0.001,
    # 0.01,
    # 0.1,
]

class_weight_list = [
    0.01,
    0.1,
    1,
]

membership_weight_list = [
    0.01,
    0.1,
    1,
]

beta_list = [
    0.0001,
    0.001,
    0.01,
    0.1,
    # 1.0,
]

repeat_idx_list = [
    0,
    # 1,
    # 2,
    # 3,
    # 4,
]

setup_dict = {
    'train_reconstructor': '1',
    'reconstruct_datasets': '1',

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 40,
    'gpu_id': 3,
    'print_training': False,
}

if not os.path.exists('log'):
    os.mkdir('log')
f = open('log/reconstruction_run.log', 'a')
f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

for dataset in dataset_list:
    for reconstruction_model in reconstruction_model_list:
        for beta in beta_list:
            for z_dim in z_dim_list:
                for architecture in architecture_list:
                    for disentanglement_type in disentanglement_type_list:
                        for setsize in setsize_list:
                            for lr in lr_list:
                                for ref_ratio in ref_ratio_list:
                                    for class_weight in class_weight_list:
                                        for membership_weight in membership_weight_list:
                                            for repeat_idx in repeat_idx_list:
                                                args_list = []
                                                target_setup_dict = setup_dict
                                                target_setup_dict['dataset'] = dataset
                                                target_setup_dict['reconstruction_model'] = reconstruction_model
                                                target_setup_dict['z_dim'] = z_dim
                                                target_setup_dict['architecture'] = architecture
                                                target_setup_dict['disentanglement_type'] = disentanglement_type
                                                target_setup_dict['setsize'] = setsize
                                                target_setup_dict['repeat_idx'] = str(repeat_idx)
                                                target_setup_dict['lr'] = str(lr)
                                                target_setup_dict['ref_ratio'] = str(ref_ratio)
                                                target_setup_dict['class_weight'] = str(class_weight)
                                                target_setup_dict['membership_weight'] = str(membership_weight)
                                                target_setup_dict['beta'] = str(beta)
                                                args_list.append('python reconstruction_main.py')
                                                for k, v in target_setup_dict.items():
                                                    args_list.append('--{} {}'.format(k, v))

                                                model = ' '.join(args_list)
                                                print(model)
                                                f.write(model + '\n')
                                                os.system(model)

f.write('\n')
f.close()
