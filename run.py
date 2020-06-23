import os

dataset_list = [
    'CIFAR-10',
]

classification_model_list = [
    # 'VGG19',
    'ResNet18',
]

setsize_list = [
    # 1000,
    10000,
    # 20000,
]

disentanglement_type = [
    'base',
    # 'type1',
    # 'type2',
]

repeat_idx_list = [
    0,
    1,
    2,
    3,
    4,
]

setup_dict = {
    'train_classifier': '1',
    'test_classifier': '1',
    'extract_classifier_features': '1',

    'statistical_attack': '0',

    'attack_type': 'black',
    'train_attacker': '0',
    'test_attacker': '0',

    'train_disentangler': '0',

    'z_dim': 64,

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 0,
}

for dataset in dataset_list:
    for classification_model in classification_model_list:
        for setsize in setsize_list:
            for repeat_idx in repeat_idx_list:
                args_list = []
                target_setup_dict = setup_dict
                target_setup_dict['dataset'] = dataset
                target_setup_dict['classification_model'] = classification_model
                target_setup_dict['setsize'] = setsize
                target_setup_dict['repeat_idx'] = str(repeat_idx)

                args_list.append('python main_for_run.py')
                for k, v in target_setup_dict.items():
                    args_list.append('--{} {}'.format(k, v))

                model = ' '.join(args_list)
                print(model)
                os.system(model)
