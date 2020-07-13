import os

dataset_list = [
    # 'adult',
    'location',
    # 'MNIST',
    # 'Fashion-MNIST',
    # 'CIFAR-10',
    # 'CIFAR-100',
]

classification_model_list = [
    'FCNClassifier',
    # 'ConvClassifier',
    # 'VGG19',
    # 'ResNet18',
    # 'ResNet50',
    # 'ResNet101',
    # 'DenseNet121',
]

setsize_list = [
    # 100,
    # 200,
    # 300,
    # 400,
    # 500,
    # 1000,
    2000,
    # 10000,
    # 20000,
    # 30000,
    # 30000,
]

target_data_list = [
    'original',
    # 'AE_z64_base',
    # 'AE_z64_type1',
    # 'AE_z64_type2',
    # 'AE_z32_base',
    # 'AE_z32_type1',
    # 'AE_z32_type2',
    # 'AE_z32_type3',
    # 'AE_z32_type4',
    # 'AE_z32_type5',
    # 'AE_z64_base',
    # 'AE_z64_type1',
    # 'AE_z64_type2',
    # 'AE_z64_type5',
    # 'AE_z16_type5',
]

recon_type_list = [
    'full_z',
    # 'content_z',
    # 'style_z',
]

repeat_idx_list = [
    0,
    # 1,
    # 2,
    # 3,
    # 4,
]

setup_dict = {
    'train_classifier': '1',
    'test_classifier': '1',
    'extract_classifier_features': '1',

    # 'recon_type': 'full_z',
    # 'recon_type': 'partial_z',

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 0,
}

for dataset in dataset_list:
    for classification_model in classification_model_list:
        for setsize in setsize_list:
            for target_data in target_data_list:
                for recon_type in recon_type_list:
                    for repeat_idx in repeat_idx_list:
                        args_list = []
                        target_setup_dict = setup_dict
                        target_setup_dict['dataset'] = dataset
                        target_setup_dict['classification_model'] = classification_model
                        target_setup_dict['setsize'] = setsize
                        target_setup_dict['target_data'] = target_data
                        target_setup_dict['recon_type'] = recon_type

                        target_setup_dict['repeat_idx'] = str(repeat_idx)

                        args_list.append('python classification_main.py')
                        for k, v in target_setup_dict.items():
                            args_list.append('--{} {}'.format(k, v))

                        model = ' '.join(args_list)
                        print(model)
                        os.system(model)
