
import numpy as np
import random

def prep_albania(selected_pixels, dataset_train_split=0.70, test=False, from_dictionary=False):

    if not from_dictionary:
        if not test:
            random.shuffle(selected_pixels)
            train_size = int(np.ceil(len(selected_pixels) * dataset_train_split))
            selected_pixels_train = selected_pixels[:train_size]
            selected_pixels_val = selected_pixels[train_size:]

            pixels_coords_train = selected_pixels_train
            samples_coords_train = [' '.join((str(x[0]), str(x[1]))) for x in pixels_coords_train]
            labels_train = [1 if x[-2] == 'g' else 0 for x in pixels_coords_train]
            patch_path_train = [x[-1] for x in pixels_coords_train]

            pixels_coords_val = selected_pixels_val
            samples_coords_val = [' '.join((str(x[0]), str(x[1]))) for x in pixels_coords_val]
            labels_val = [1 if x[-2] == 'g' else 0 for x in pixels_coords_val]
            patch_path_val = [x[-1] for x in pixels_coords_val]

            return samples_coords_train, labels_train, patch_path_train, samples_coords_val, labels_val, patch_path_val
        else:
            random.shuffle(selected_pixels)
            pixels_coords_test = selected_pixels
            samples_coords_test = [' '.join((str(x[0]), str(x[1]))) for x in pixels_coords_test]
            labels_test= [1 if x[-2] == 'g' else 0 for x in pixels_coords_test]
            patch_path_test = [x[-1] for x in pixels_coords_test]

            return samples_coords_test, labels_test, patch_path_test

    else:
        if not test:
            tot_len = len(selected_pixels['patches'])
            train_len = int(tot_len * dataset_train_split)
            #val_len = int(tot_len * 0.33)

            items = list(zip(selected_pixels['patches'], selected_pixels['labels']))
            random.shuffle(items)

            patches, labels = zip(*items)
            labels = list(labels)
            patches_train = patches[:train_len]
            patches_val = patches[train_len:tot_len]
            labels_train = labels[:train_len]
            labels_val = labels[train_len:tot_len]

            train_dict = {}
            val_dict = {}

            train_dict['patches'] = patches_train
            train_dict['labels'] = labels_train
            val_dict['patches'] = patches_val
            val_dict['labels'] = labels_val

            return train_dict, val_dict

        else:

            tot_len = len(selected_pixels['patches'])
            items = list(zip(selected_pixels['patches'], selected_pixels['labels']))
            random.shuffle(items)

            patches, labels = zip(*items)
            labels = list(labels)
            patches_train = patches[:dataset_train_split]
            patches_val = patches[dataset_train_split:tot_len]
            labels_train = labels[:dataset_train_split]
            labels_val = labels[dataset_train_split:tot_len]

            train_dict = {}
            val_dict = {}

            train_dict['patches'] = patches_train
            train_dict['labels'] = labels_train
            val_dict['patches'] = patches_val
            val_dict['labels'] = labels_val

            return val_dict
