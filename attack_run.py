import os
import sys
import datetime

dataset_list = [
    'adult',
    # 'CIFAR-10',
]

target_classifier_list = [
    # 'ResNet18_setsize1000_original',
    # 'ResNet18_setsize1000_AE_z64_base/full_z',

    # 'ResNet18_setsize10000_original',
    # 'ResNet18_setsize10000_AE_z64_base/full_z',

    # 'ResNet50_setsize10000_original',
    # 'ResNet50_setsize10000_AE_z64_base/partial_z',
    # 'ResNet50_setsize10000_AE_z64_base/full_z',
    # 'ResNet50_setsize10000_AE_z64_type1/partial_z',
    # 'ResNet50_setsize10000_AE_z64_type2/full_z',
    # 'ResNet50_setsize10000_AE_z64_type2/partial_z',

    # 'ResNet50_setsize20000_original',

    # 'ResNet101_setsize10000_original',
    # 'ResNet101_setsize20000_original',

    'FCN_setsize100_AE_z8_base/partial_z',
    'FCN_setsize100_AE_z8_type1/partial_z',
    'FCN_setsize100_AE_z8_type2/partial_z',
]

repeat_idx_list = [
    0,
#     1,
#     2,
#     3,
#     4,
]

setup_dict = {
    'train_attacker': '1',
    'test_attacker': '1',
    'attack_type': 'black',

    'statistical_attack': '1',

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 0,
}

if not os.path.exists('log'):
    os.mkdir('log')
f = open('log/attack_run.log', 'a')
f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

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
            f.write(model + '\n')
            os.system(model)

f.write('\n')
f.close()
