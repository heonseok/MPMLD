import os
import sys
import datetime

dataset_list = [
    # 'adult',
    'location',
    # 'MNIST',
    # 'Fashion-MNIST',
    # 'CIFAR-10',
]

target_classifier_list = [
    # 'original_setsize2000_FCNClassifierA',

    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.01_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw0.1_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.01_mw1.0_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.01_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw0.1_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.01_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw0.1_FCNClassifierA',
    # 'AE_z64_setsize2000_lr0.001_ref0.1_arcC_type5_cw1.0_mw1.0_FCNClassifierA',

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0_FCNClassifierA',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0_FCNClassifierA',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0_FCNClassifierA',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0_FCNClassifierA',

    # 'VAE0.0001_z64_setsize1000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'VAE0.001_z64_setsize1000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0_FCNClassifierA',

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0_FCNClassifierA',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0_FCNClassifierA',

    # 'VAE0.0001_z64_setsize2000_lr0.001_ref0.1_arcD_base_FCNClassifierA',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_base_FCNClassifierA',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.01_mw1.0_FCNClassifierA',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'VAE1e-06_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0_FCNClassifierA',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0_FCNClassifierA',

    # 0722
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_type5_cw1.0_mw10.0_FCNClassifierA',
    # 'VAE0.001_z64_setsize2000_lr0.001_ref0.1_arcD_base_FCNClassifierA',
    # 'VAE0.001_z64_setsize500_lr0.001_ref0.1_arcD_base_FCNClassifierA',
    # 'VAE0.001_z64_setsize1000_lr0.001_ref0.1_arcD_base_FCNClassifierA',
    # 'original_setsize500_FCNClassifierA',
    # 'original_setsize1000_FCNClassifierA',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_base_FCNClassifierA',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw1.0_mw10.0_FCNClassifierA',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw1.0_mw20.0_FCNClassifierA',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw1.0_mw1.0_FCNClassifierA',
    # 'VAE0.001_z128_setsize1000_lr0.001_ref0.1_arcD_type5_cw10.0_mw1.0_FCNClassifierA',

    # 0723
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_base_FCNClassifierA',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw10.0_FCNClassifierA',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw10.0_mw10.0_FCNClassifierA',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw1.0_FCNClassifierA',
    # 'VAE0.001_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw0.1_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw1.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw2.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw5.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw2.0_mw1.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw1.0_mw0.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.1_mw0.1_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.01_mw0.1_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.01_mw1.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref0.1_arcE_type5_cw0.01_mw5.0_FCNClassifierA',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw1.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw0.1_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw1.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.01_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.001_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.005_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.0_mw10.0_FCNClassifierA', # recon weight 10
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.01_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw10.0_FCNClassifierA',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.02_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.03_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.04_mw10.0_FCNClassifierA',
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.05_mw10.0_FCNClassifierA',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.011_mw10.0_FCNClassifierA',

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.1_mw10.0_FCNClassifierA',  # recon weight 100
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw10.0_FCNClassifierA',  # recon weight 100
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw0.0_mw10.0_FCNClassifierA',  # recon weight 100

    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw0.0_FCNClassifierA',  # recon weight 100
    # 'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw0.01_FCNClassifierA',  # recon weight 100
    'VAE1e-05_z256_setsize1000_lr0.001_ref1.0_arcE_type5_cw1.0_mw0.1_FCNClassifierA',  # recon weight 100
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
    'train_attacker': '1',
    'test_attacker': '1',
    'attack_type': 'black',

    'statistical_attack': '1',

    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 3,
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

            if 'original' in target_classifier:
                for k, v in target_setup_dict.items():
                    args_list.append('--{} {}'.format(k, v))

                model = ' '.join(args_list)
                print(model)
                f.write(model + '\n')
                os.system(model)
                continue

            for recon_type in recon_type_list:
                target_setup_dict['recon_type'] = recon_type
                for k, v in target_setup_dict.items():
                    args_list.append('--{} {}'.format(k, v))

                model = ' '.join(args_list)
                print(model)
                f.write(model + '\n')
                os.system(model)

f.write('\n')
f.close()
