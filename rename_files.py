import os
import sys

base_path = '/mnt/disk1/heonseok/MPMLD/SVHN/0825_4typesDisentanglement_small_recon'
for model_name in os.listdir(base_path):
    # print(model_name)
    if 'Disc' not in model_name:
        print(model_name)
        # elements = model_name.split('_')
        # elements = elements[:2] + ['distinctDisc'] + elements[2:]
        # new_model_name = ''
        # for idx, element in enumerate(elements):
        #     new_model_name += element
        #     if idx < len(elements)-1:
        #         new_model_name += '_'
        # print(new_model_name)
        #
        # os.rename(os.path.join(base_path, model_name), os.path.join(base_path, new_model_name))


