import numpy as np
import os


roi_path = '../algonauts_2023_challenge_data/subj01/roi_masks/'

streams = np.load(roi_path + 'lh.streams_challenge_space.npy', allow_pickle=True)
mapping = np.load(roi_path + 'mapping_streams.npy', allow_pickle=True).item()

inverse_mapping = {name: val for val, name in mapping.items()}
ventral = np.where(streams == inverse_mapping['midventral'])[0]
print(mapping)
print(ventral, ventral.shape)
x = {}