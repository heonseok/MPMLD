import os

dataset_list = [
    'CIFAR-10',
]

target_classifier_list = [
    # 'ResNet18_setsize1000',
    'ResNet18_setsize20000',
    # 'ResNet18_setsize30000',
]

repeat_idx_list = [
    0,
    1,
    2,
    3,
    4,
]

setup_dict = {
    'train_attacker': '1',
    'test_attacker': '1',
    'attack_type': 'black',

    'statistical_attack': '1',

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 1,
}

for dataset in dataset_list:
    for target_classifier in target_classifier_list:
        for repeat_idx in repeat_idx_list:
            args_list = []
            target_setup_dict = setup_dict
            target_setup_dict['dataset'] = dataset
            target_setup_dict['target_classifier'] = target_classifier
            target_setup_dict['repeat_idx'] = str(repeat_idx)

            args_list.append('python attack_main.py')
            for k, v in target_setup_dict.items():
                args_list.append('--{} {}'.format(k, v))

            model = ' '.join(args_list)
            print(model)
            os.system(model)
