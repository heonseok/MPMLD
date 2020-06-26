import os
import sys
import datetime

dataset_list = [
    'CIFAR-10',
]

setsize_list = [
    # 1000,
    # 10000,
    20000,
]

reconstruction_model_list = [
    # 'AE',
    'VAE'
]

disentanglement_type_list = [
    'base',
    # 'type1',
    # 'type2',
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

    'z_dim': 64,

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 50,
    'gpu_id': 0,

    'lr': 0.002,
}

if not os.path.exists('log'):
    os.mkdir('log')
f = open('log/reconstruction_run.log', 'a')
f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

for dataset in dataset_list:
    for reconstruction_model in reconstruction_model_list:
        for disentanglement_type in disentanglement_type_list:
            for setsize in setsize_list:
                for repeat_idx in repeat_idx_list:
                    args_list = []
                    target_setup_dict = setup_dict
                    target_setup_dict['dataset'] = dataset
                    target_setup_dict['reconstruction_model'] = reconstruction_model
                    target_setup_dict['disentanglement_type'] = disentanglement_type
                    target_setup_dict['setsize'] = setsize
                    target_setup_dict['repeat_idx'] = str(repeat_idx)

                    args_list.append('python reconstruction_main.py')
                    for k, v in target_setup_dict.items():
                        args_list.append('--{} {}'.format(k, v))

                    model = ' '.join(args_list)
                    print(model)
                    f.write(model + '\n')
                    os.system(model)

f.write('\n')
f.close()
