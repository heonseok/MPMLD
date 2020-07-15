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

classifier_type_list = [
    'A',
    # 'B',
]

target_data_list = [
    # 'original_setsize2000',

    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcA_type5_cw0.01_mw1.0',
    # 'AE_z64_setsize2000_lr0.01_ref0.1_arcA_type5_cw0.01_mw1.0',
    # 'AE_z64_setsize2000_lr0.1_ref0.1_arcA_type5_cw0.01_mw1.0',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcA_type5_cw1.0_mw0.01',
    # 'AE_z64_setsize2000_lr0.01_ref0.1_arcA_type5_cw1.0_mw0.01',
    # 'AE_z64_setsize2000_lr0.1_ref0.1_arcA_type5_cw1.0_mw0.01',

    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.01',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.1',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw1.0',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.01',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.1',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw1.0',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.01',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.1',
    'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw1.0',

]

recon_type_list = [
    'full_z',
    'content_z',
    'style_z',
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

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 3,
}

for dataset in dataset_list:
    for classification_model in classification_model_list:
        for classifier_type in classifier_type_list:
            for repeat_idx in repeat_idx_list:
                for target_data in target_data_list:
                    args_list = []
                    target_setup_dict = setup_dict
                    target_setup_dict['dataset'] = dataset
                    target_setup_dict['classification_model'] = classification_model
                    target_setup_dict['target_data'] = target_data
                    target_setup_dict['classifier_type'] = classifier_type
                    target_setup_dict['repeat_idx'] = str(repeat_idx)

                    args_list.append('python classification_main.py')

                    if 'original' in target_data:
                        for k, v in target_setup_dict.items():
                            args_list.append('--{} {}'.format(k, v))

                        model = ' '.join(args_list)
                        print(model)
                        os.system(model)
                        continue

                    for recon_type in recon_type_list:
                        target_setup_dict['recon_type'] = recon_type
                        for k, v in target_setup_dict.items():
                            args_list.append('--{} {}'.format(k, v))

                        model = ' '.join(args_list)
                        print(model)
                        os.system(model)
