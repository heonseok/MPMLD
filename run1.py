import os

dataset_list = [
    # 'adult'
    # 'location',
    # 'MNIST',
    # 'Fashion-MNIST',
    'SVHN',
    # 'CIFAR-10',
]

setsize_list = [
    # 100,
    # 500,
    # 1000,
    # 2000,
    # 4000,
    5000,
    # 5000,
    # 10000,
    # 20000,
]

z_dim_list = [
    # '16',
    # '32',
    # '64',
    '128',
    # '256',
]

ref_ratio_list = [
    # 0.1,
    # 0.2,
    # 0.5,
    1.0,
    # 2.0,
]

recon_lr_list = [
    # 0.0001,
    0.001,
    # 0.01,
    # 0.1,
]

# recon, class_pos, class_neg, membership_pos, membership_neg
weight_list = [
    # [1, 0, 0, 0, 0],
    # [10, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    # [1, 10, 10, 10, 10],
]

beta_list = [
    # 0.0
    # 0.000001,
    # 0.00001,
    # 0.0001,
    # 0.001,
    0.01,
    # 0.05,
    # 0.1,
    # 1.0,
]

setup_dict = {
    'base_path': '/mnt/disk1/heonseok/MPMLD',

    # Reconstruction
    'share_encoder': '0',
    'share_discriminator': '1',
    'recon_train_batch_size': 32,
    'train_reconstructor': '0',
    'reconstruct_datasets': '0',
    'plot_recons': '1',

    'use_reconstructed_dataset': '1',
    'disentangle_with_reparameterization': '1',

    # Classification
    'class_train_batch_size': 32,
    'train_classifier': '0',
    'test_classifier': '0',
    'extract_classifier_features': '0',
    'classification_model': 'ResNet18',
    # 'classification_model': 'FCClassifier',

    # Attack
    'train_attacker': '0',
    'test_attacker': '0',

    # Common
    'repeat_start': 0,
    'repeat_end': 5,
    'epochs': 500,
    'early_stop': '1',
    'early_stop_observation_period': 20,
    'gpu_id': 3,
    'print_training': '0',
    'description': '0825_4typesDisentanglement_small_recon',
}

for dataset in dataset_list:
    for beta in beta_list:
        for z_dim in z_dim_list:
            for setsize in setsize_list:
                for recon_lr in recon_lr_list:
                    for ref_ratio in ref_ratio_list:
                        for weight in weight_list:
                            args_list = []
                            target_setup_dict = setup_dict
                            target_setup_dict['dataset'] = dataset
                            target_setup_dict['z_dim'] = z_dim
                            target_setup_dict['setsize'] = setsize
                            target_setup_dict['recon_lr'] = str(recon_lr)
                            target_setup_dict['ref_ratio'] = str(ref_ratio)
                            target_setup_dict['beta'] = str(beta)

                            target_setup_dict['recon_weight'] = str(weight[0])
                            target_setup_dict['class_pos_weight'] = str(weight[1])
                            target_setup_dict['class_neg_weight'] = str(weight[2])
                            target_setup_dict['membership_pos_weight'] = str(weight[3])
                            target_setup_dict['membership_neg_weight'] = str(weight[4])

                            args_list.append('python main.py')
                            for k, v in target_setup_dict.items():
                                args_list.append('--{} {}'.format(k, v))

                            model = ' '.join(args_list)
                            print(model)
                            print()
                            os.system(model)
