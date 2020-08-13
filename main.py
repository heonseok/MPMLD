import os
import sys
import argparse
from utils import str2bool
import datetime
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils
from data import load_dataset

import torch
from torch.utils.data import Subset, ConcatDataset

# from reconstruction import Reconstructor
# from reconstruction_stylez import Reconstructor
from reconstruction_class_conditional import Reconstructor
from classification import Classifier
from attack import Attacker

parser = argparse.ArgumentParser()

# -------------------------------------------------------------------------------------------------------------------- #
# -------- Params -------- #
# ---- Common ---- #
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--dataset', type=str, default='SVHN',
                    choices=['MNIST', 'Fashion-MNIST', 'SVHN', 'CIFAR-10', 'adult', 'location', ])
parser.add_argument('--setsize', type=int, default=5000)
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--print_training', type=str2bool, default='0')

# ---- Reconstruction ---- #
parser.add_argument('--reconstruction_model', type=str, default='VAE', choices=['AE', 'VAE'])
parser.add_argument('--beta', type=float, default=0.000001)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--recon_lr', type=float, default=0.001)
parser.add_argument('--disc_lr', type=float, default=0.001)

parser.add_argument('--recon_weight', type=float, default='1')
parser.add_argument('--class_cz_weight', type=float, default='0')
parser.add_argument('--class_mz_weight', type=float, default='0')
parser.add_argument('--membership_cz_weight', type=float, default='1')
parser.add_argument('--membership_mz_weight', type=float, default='1')
parser.add_argument('--ref_ratio', type=float, default=1.0)

# ---- Classification ---- #
parser.add_argument('--classification_model', type=str, default='ResNet18',
                    choices=['FCClassifier', 'ConvClassifier', 'VGG19', 'ResNet18', 'ResNet50', 'ResNet101',
                             'DenseNet121'])
parser.add_argument('--class_lr', type=float, default=0.0001)

# -------- Attack -------- #
parser.add_argument('--attack_lr', type=float, default=0.01)

# -------------------------------------------------------------------------------------------------------------------- #
# -------- Control flags -------- #
parser.add_argument('--description', type=str, default='0805conditional_membership_disc')
# ---- Reconstruction ---- #
parser.add_argument('--train_reconstructor', type=str2bool, default='1')
parser.add_argument('--reconstruct_datasets', type=str2bool, default='1')
parser.add_argument('--plot_recons', type=str2bool, default='1')

# ---- Classification ---- #
parser.add_argument('--use_reconstructed_dataset', type=str2bool, default='1')

parser.add_argument('--train_classifier', type=str2bool, default='1')
parser.add_argument('--test_classifier', type=str2bool, default='1')
parser.add_argument('--extract_classifier_features', type=str2bool, default='1')

# ---- Attack ---- #
parser.add_argument('--train_attacker', type=str2bool, default='1')
parser.add_argument('--test_attacker', type=str2bool, default='1')

# -------------------------------------------------------------------------------------------------------------------- #

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# ---- Directory ---- #
if args.reconstruction_model == 'VAE':
    args.reconstruction_model += str(args.beta)

args.reconstruction_name = os.path.join(
    '{}_z{}_setsize{}_lr{}_ref{}_rw{}_cc{}_cm{}_mc{}_mm{}'.format(args.reconstruction_model, args.z_dim, args.setsize,
                                                                  args.recon_lr, args.ref_ratio, args.recon_weight,
                                                                  args.class_cz_weight, args.class_mz_weight,
                                                                  args.membership_cz_weight, args.membership_mz_weight,
                                                                  ))

args.recon_output_path = os.path.join(args.base_path, args.dataset, args.description, args.reconstruction_name)
args.raw_output_path = os.path.join(args.base_path, args.dataset, 'raw_setsize{}'.format(args.setsize))

if not os.path.exists(args.recon_output_path):
    os.makedirs(args.recon_output_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

args.model_path = os.path
args.reconstruction_path = os.path.join(args.recon_output_path, 'reconstruction/repeat{}'.format(args.repeat_idx))
print(args.reconstruction_path)

# ---- Backup codes ---- #
date = str(datetime.datetime.now())[:-16]
time = str(datetime.datetime.now())[-15:-7]
backup_path = os.path.join('backup', date, time + ' ' + args.reconstruction_name)
os.makedirs(backup_path)
for file in os.listdir(os.getcwd()):
    if file.endswith('.py'):
        shutil.copy2(file, backup_path)

# -------------------------------------------------------------------------------------------------------------------- #
# ---- Dataset ---- #
merged_dataset = load_dataset(args.dataset, args.data_path)
print(merged_dataset.__len__())

if args.setsize * 2.4 > len(merged_dataset):
    print('Setsize * 2.4 > len(merged_dataset); Terminate program')
    sys.exit(1)

if args.dataset in ['adult', 'location']:
    args.encoder_input_dim = merged_dataset.__getitem__(0)[0].numpy().shape[0]
    if args.dataset == 'adult':
        args.class_num = 2
    elif args.dataset == 'location':
        args.class_num = 30

elif args.dataset in ['MNIST', 'SVHN', 'CIFAR-10']:
    args.class_num = 10

# Recon: Train, Class: Train, Attack: In(Train/Test)
subset0 = Subset(merged_dataset, range(0, args.setsize))
# Recon: Valid, Class: Valid, Attack: -
subset1 = Subset(merged_dataset, range(args.setsize, int(1.2 * args.setsize)))
# Recon: Test, Class: Test, Attack: -
subset2 = Subset(merged_dataset, range(int(1.2 * args.setsize), int(1.4 * args.setsize)))
# Recon: -, Class: -,  Attack: Out(Train/Test) Todo: check valid
subset3 = Subset(merged_dataset, range(int(1.4 * args.setsize), int(2.4 * args.setsize)))
# Recon: Train (Reference), Class: -, Attack: -
subset4 = Subset(merged_dataset, range(int(2.4 * args.setsize), int((2.4 + args.ref_ratio) * args.setsize)))

class_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}

for dataset_type, dataset in class_datasets.items():
    print('Class {:<5} dataset: {}'.format(dataset_type, len(dataset)))
print()

# ---- Combination ---- #
reconstruction_type_list = [
    # 'cb_mb_sb',  # Content: base, Membership: base, Style: base
    # 'cb_mb_sz',  # Content: base, Membership: base, Style: zero
    # 'cb_mz_sb',  # Content: base, Membership: zero, Style: base
    # 'cb_mz_sz',  # Content: base, Membership: zero, Style: zero
    # 'cz_mb_sb',  # Content: zero, Membership: base, Style: base
    # 'cz_mb_sz',  # Content: zero, Membership: base, Style: zero

    # 'cb_mr',
    # 'cs1.2_mb0.8',
    # 'cb_mb_n1',
    # 'cb_mb_n0.5',
    # 'cb_mb_n0.1',
    # 'cb_ms0.8',  # Content: base, Membership: scaled
    # 'cb_ms0.8_n0.2',  # Content: base, Membership: scaled
    # 'cb_ms0.5',  # Content: base, Membership: scaled
    # 'cb_ms0.25',  # Content: base, Membership: scaled
    # 'cb_ms0.1',  # Content: base, Membership: scaled
    # 'cb_mz',  # Content: base, Membership: zero
    'cb_mb',  # Content: base, Membership: base
    # 'cz_mb',  # Content: zero, Membership: base
    # 'cb_ms0.5_n0.5',
    # 'cb_ms0.5_n0.1',
    # 'cb_ms',  # Content: base, Membership: sampled
    # 'cs_mb',  # Content: sampled, Membership: zero
    # 'cs_ms',  # Content: sampled, Membership: sampled
    # 'cb_mn',  # Content: base, Membership: normal

    # 'cb_mConstant',
    # 'cb_mConstant0.8',
    # 'cb_mInter0.8',
    # 'cb_mr1.2',
    # 'cb_mr2.0',
    # 'cb_mAvg',
]

attack_type_list = [
    'black',
    # 'white',
]

# -------------------------------------------------------------------------------------------------------------------- #
# -------- Reconstruction -------- #
if args.train_reconstructor:
    reconstructor = Reconstructor(args)
    ref_dataset = subset4
    for _ in range(int(1 / args.ref_ratio) - 1):
        ref_dataset = ConcatDataset((ref_dataset, subset4))
    reconstructor.train(class_datasets['train'], class_datasets['valid'], ref_dataset)

if args.reconstruct_datasets:
    reconstructor = Reconstructor(args)
    inout_datasets = {
        'train': subset0,  # todo : rename train --> in (?)
        'out': subset3,
    }
    reconstructor.reconstruct(inout_datasets, reconstruction_type_list)
    # plot --> rclone

if args.plot_recons:
    # img_path = '{}_{}_{}_{}.png'.format(args.dataset, args.description, args.reconstruction_name, args.repeat_idx)
    img_dir = os.path.join('Figs', args.dataset, args.description)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_list = ['raw.png']
    for recon_type in reconstruction_type_list:
        img_list.append(recon_type + '.png')
    # img_list = [
    #     'raw.png',
    #     'cb_mb_n1.png',
    #
    #     # 'cb_ms0.8.png',
    #     # 'cb_ms0.5.png',
    #     # 'cb_ms0.25.png',
    #     # 'cb_ms0.1.png',
    #     # 'cb_mz.png',
    #     # 'cb_mb.png',
    #     # 'cz_mb.png',
    #
    #     # 'cb_mb_sb.png',
    #     # 'cb_mb_sz.png',
    #     # 'cb_mz_sb.png',
    #     # 'cb_mz_sz.png',
    #     # 'cz_mb_sb.png',
    #     # 'cz_mb_sz.png',
    # ]

    plt.figure(1, figsize=(10, 4))
    for img_idx, recon_type in enumerate(img_list):
        plt.subplot(str(1) + str(len(img_list)) + str(img_idx + 1))
        plt.imshow(mpimg.imread(os.path.join(args.reconstruction_path, recon_type)))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    img_path = os.path.join(img_dir, '{}_repeat{}.png'.format(args.reconstruction_name, args.repeat_idx))
    plt.savefig(img_path)

    drive_path = os.path.join('Research/MPMLD/', img_dir)
    os.system('rclone copy -P {} remote:{}'.format(img_path, drive_path))

print()

# -------- Classification & Attack -------- #
if args.use_reconstructed_dataset:
    for recon_type in reconstruction_type_list:
        args.classification_path = os.path.join(args.recon_output_path, 'classification', args.classification_model,
                                                recon_type, 'repeat{}'.format(args.repeat_idx))
        # print(args.classification_path)
        if args.train_classifier or args.test_classifier or args.extract_classifier_features:
            classifier = Classifier(args)

            try:
                reconstructed_data_path = os.path.join(args.reconstruction_path, 'recon_{}.pt'.format(recon_type))
                recon_datasets = utils.build_reconstructed_datasets(reconstructed_data_path)
                class_datasets['train'] = recon_datasets['train']
            except FileNotFoundError:
                print('There is no reconstructed data: ', args.reconstruction_path)
                sys.exit(1)

            if args.train_classifier:
                classifier.train(class_datasets['train'], class_datasets['valid'])

            if args.test_classifier:
                classifier.test(class_datasets['test'])

            if args.extract_classifier_features:
                # inout_datasets should be transformed to inout_feature_sets for training attacker
                inout_datasets = {
                    'in': subset0,
                    'out': subset3,
                }

                # for dataset_type, dataset in inout_datasets.items():
                #     print('Inout {:<3} dataset: {}'.format(dataset_type, len(dataset)))
                # print()
                classifier.extract_features(inout_datasets)

        if args.train_attacker or args.test_attacker:
            for attack_type in attack_type_list:
                args.attack_type = attack_type
                args.attack_path = os.path.join(args.recon_output_path, 'attack', args.classification_model,
                                                recon_type, attack_type, 'repeat{}'.format(args.repeat_idx))
                if not os.path.exists(args.attack_path):
                    os.makedirs(args.attack_path)

                inout_feature_sets = utils.build_inout_feature_sets(args.classification_path, attack_type)

                # for dataset_type, dataset in inout_feature_sets.items():
                #     print('Inout {:<3} feature set: {}'.format(dataset_type, len(dataset)))

                attacker = Attacker(args)
                if args.train_attacker:
                    attacker.train(inout_feature_sets['train'], inout_feature_sets['valid'])
                if args.test_attacker:
                    attacker.test(inout_feature_sets['test'])
        print()

# else:
#     args.classification_path = os.path.join(args.raw_output_path, 'classification', args.classification_model,
#                                             'repeat{}'.format(args.repeat_idx))
#     print(args.classification_path)
#     classifier = Classifier(args)
#
#     # todo : refactor
#     if args.train_classifier:
#         classifier.train(class_datasets['train'], class_datasets['valid'])
#
#     if args.test_classifier:
#         classifier.test(class_datasets['test'])
#
#     if args.extract_classifier_features:
#         # inout_datasets should be transformed to inout_feature_sets for training attacker
#         inout_datasets = {
#             'in': subset0,
#             'out': subset3,
#         }
#
#         for dataset_type, dataset in inout_datasets.items():
#             print('Inout {:<3} dataset: {}'.format(dataset_type, len(dataset)))
#         print()
#         classifier.extract_features(inout_datasets)
