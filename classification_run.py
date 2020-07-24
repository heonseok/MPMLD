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

    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.01',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.1',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw1.0',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.01',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.1',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw1.0',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.01',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.1',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw1.0',

    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_base',

    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.01',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.1',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw0.01_mw1.0',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.01',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.1',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw0.1_mw1.0',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.01',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.1',
    # 'AE_z64_setsize1000_lr0.001_ref0.1_arcC_type5_cw1.0_mw1.0',

    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.01',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.1',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw0.01_mw1.0',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.01',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.1',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw0.1_mw1.0',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.01',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.1',
    # 'AE_z64_setsize500_lr0.001_ref0.1_arcC_type5_cw1.0_mw1.0',

    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw0.01',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw0.1',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw0.01',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw0.1',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw0.01',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw0.1',
    # 'VAE0.1_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw0.01',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw0.1',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw0.01',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw0.1',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw0.01',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw0.1',
    # 'VAE1.0_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0'

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',

    # 'VAE0.0001_z64_setsize1000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.001_z64_setsize1000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',

    # 'VAE1e-05_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE1e-05_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE1e-05_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_base',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_base',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',

    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw100.0_mw1.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',

    # 0722
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw10.0',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_base',
    # 'VAE0.001_z64_setsize500_lr0.001_ref0.1_arcD_base',
    # 'VAE0.001_z64_setsize1000_lr0.001_ref0.1_arcD_base',
    # 'original_setsize500',
    # 'original_setsize1000',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_base',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw1.0_mw10.0',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw1.0_mw20.0',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw10.0_mw1.0',

    # 0723
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw10.0_mw10.0',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcE_base',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_base',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw10.0',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw10.0_mw10.0',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw1.0',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw0.1',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw1.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw2.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw5.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw2.0_mw1.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw0.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.1_mw0.1',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.01_mw0.1',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.01_mw1.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.01_mw5.0',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw1.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw0.1',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw1.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.01_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.001_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.005_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.0_mw10.0', # recon weight 10
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.01_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw10.0',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.02_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.03_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.04_mw10.0',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.05_mw10.0',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.011_mw10.0',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw10.0',  # recon weight 100
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw10.0',  # recon weight 100
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.0_mw10.0',  # recon weight 100

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw0.0',  # recon weight 100
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw0.01',  # recon weight 100
    'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw0.1',  # recon weight 100
]

recon_type_list = [
    'base_z',
    # 'content_z',
    # 'style_z',
    # 'full_z',
    'zero_content',
    'zero_style',
    # 'uniform_style',
    # 'normal_style',
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

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 3,
    'print_training': False,
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
                        # todo : fix duplication
                        target_setup_dict['recon_type'] = recon_type
                        for k, v in target_setup_dict.items():
                            args_list.append('--{} {}'.format(k, v))

                        model = ' '.join(args_list)
                        print(model)
                        os.system(model)
