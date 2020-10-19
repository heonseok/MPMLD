import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def collate_recon_results(recon_path, recon_dict, early_stop_flag, target_epoch):
    df = pd.DataFrame()
    for repeat_idx in range(5):
        repeat_path = os.path.join(recon_path, 'repeat{}'.format(repeat_idx))

        if not early_stop_flag:
            class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc{:03d}.npy'.format(target_epoch)), allow_pickle=True).item()
            membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc{:03d}.npy'.format(target_epoch)), allow_pickle=True).item()
        else:
            class_acc_dict = np.load(os.path.join(repeat_path, 'class_acc.npy'), allow_pickle=True).item()
            membership_acc_dict = np.load(os.path.join(repeat_path, 'membership_acc.npy'), allow_pickle=True).item()

        for z_type in ['pn', 'pp', 'np', 'nn']:
            acc_dict = {'disc_type': 'class', 'z_type': z_type, 'acc': class_acc_dict[z_type]}
            df = df.append({**recon_dict, **acc_dict}, ignore_index=True)
            acc_dict = {'disc_type': 'membership', 'z_type': z_type, 'acc': membership_acc_dict[z_type]}
            df = df.append({**recon_dict, **acc_dict}, ignore_index=True)
    
    return df 

def collate_class_results(class_path, class_dict, recon_type):
    df = pd.DataFrame()
    for repeat_idx in range(5):

        if 'raw' in class_path:
            repeat_path = os.path.join(class_path, 'repeat' + str(repeat_idx), 'acc.npy')
            recon_type = 'raw'
        else:
            repeat_path = os.path.join(class_path, recon_type, 'repeat' + str(repeat_idx), 'acc.npy')

        result_dict = np.load(repeat_path, allow_pickle=True).item()
        for dataset_type in ['train', 'valid', 'test']:
            acc_dict = {'recon_type': recon_type, 'dataset': dataset_type, 'acc': result_dict[dataset_type]}
            df = df.append({**class_dict, **acc_dict}, ignore_index=True)


    return df


def collate_attack_results(attack_path, attack_dict, recon_type, attack_type):
    df = pd.DataFrame()
    for repeat_idx in range(5):
        if 'raw' in attack_path:
            repeat_path = os.path.join(attack_path, 'repeat' + str(repeat_idx), 'acc.npy')
            recon_type = 'raw'
        else:

            repeat_path = os.path.join(attack_path, recon_type, attack_type, 'repeat' + str(repeat_idx), 'acc.npy')

            result_dict = np.load(repeat_path, allow_pickle=True).item()
            acc_dict = {'recon_type': recon_type, 'attack_type': attack_type, 'acc': result_dict['test']}
            df = df.append({**attack_dict, **acc_dict}, ignore_index=True)

    return df 

def draw_results(recon_df, class_df, attack_df, expr, column_type):
    target_recon_df = recon_df.copy()
    target_recon_df = target_recon_df.query(expr)
    sns.catplot(data=target_recon_df, x='disc_type', y='acc', hue='z_type', col=column_type, kind='box')

    target_class_df = class_df.copy()
    target_class_df = target_class_df.query(expr)
    sns.catplot(data=target_class_df, x='recon_type', y='acc', hue='dataset', col=column_type, kind='box')

    target_attack_df = attack_df.copy()
    target_attack_df = target_attack_df.query(expr)
    sns.catplot(data=target_attack_df, x='recon_type', y='acc', hue='attack_type', col=column_type, kind='box')

def collate_dfs(base_path, dataset_list, description_list, beta_list, z_dim_list, setsize_list, recon_lr_list, ref_ratio_list, weight_list, small_recon_weight_list, recon_train_batch_size, recon_type_list, early_stop_flag, target_epoch):

    recon_df = pd.DataFrame()
    class_df = pd.DataFrame()
    attack_df = pd.DataFrame()

    for dataset in dataset_list:
        for description in description_list:
            for beta in beta_list:
                for z_dim in z_dim_list:
                    for setsize in setsize_list:
                        for recon_lr in recon_lr_list:
                            for ref_ratio in ref_ratio_list:
                                for weight in weight_list:
                                    for small_recon_weight in small_recon_weight_list:

                                        reconstruction_name = '{}{}_{}Enc_{}Disc_z{}_setsize{}_lr{}_bs{}_ref{}_rw{}_rf{}_cp{}_cn{}_mp{}_mn{}_sr{}'.format(
                                            'VAE',
                                            beta,
                                            'distinct',
                                            'distinct',
                                            z_dim,
                                            setsize, 
                                            recon_lr,
                                            recon_train_batch_size,
                                            ref_ratio,
                                            weight[0],
                                            weight[1],
                                            weight[2],
                                            weight[3],
                                            weight[4],
                                            weight[5],
                                            small_recon_weight,
                                        )

                                        weight_str = '[' + ', '.join([str(elem) for elem in weight]) + ']'

                                        recon_dict = {
                                            'dataset': dataset,
                                            'description': description,
                                            'beta': float(beta), 
                                            'z_dim' : int(z_dim),
                                            'setsize' : int(setsize),
                                            'recon_lr': float(recon_lr),
                                            'recon_train_batch_size': int(recon_train_batch_size),
                                            'ref_ratio': float(ref_ratio),
                                            #    'recon_weight': weight[0],
                                            #    'realfake_weight': weight[1],
                                            #    'class_pos_weight': weight[2],
                                            #    'class_neg_weight': weight[3],
                                            #    'membership_pos_weight': weight[4],
                                            #    'membership_neg_weight': weight[5],
                                            'weights' : weight_str, 
                                            'small_recon_weight': float(small_recon_weight),
                                        }

                                        try:
                                            recon_path = os.path.join(base_path, dataset, description, reconstruction_name, 'reconstruction')
                                            recon_df = pd.concat([recon_df, collate_recon_results(recon_path, recon_dict, early_stop_flag, target_epoch)])

                                        except FileNotFoundError:
                                            pass

                                        for recon_type in recon_type_list:
                                            try:
                                                class_path = os.path.join(base_path, dataset, description, reconstruction_name, 'classification', 'ResNet18_lr0.0001_bs32')
                                                class_df = pd.concat([class_df, collate_class_results(class_path, recon_dict, recon_type)])

                                                for attack_type in ['black', 'white']:
                                                    attack_path =  os.path.join(base_path, dataset, description, reconstruction_name, 'attack', 'ResNet18_lr0.0001_bs32')
                                                    attack_df = pd.concat([attack_df, collate_attack_results(attack_path, recon_dict, recon_type, attack_type)])

                                            except FileNotFoundError:
                                                # print('There is no file : {}'.format(class_path))
                                                pass

    return recon_df, class_df, attack_df 