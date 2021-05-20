import numpy as np
import math
import torch
from torch.utils.data import Dataset

from copy import deepcopy
import pandas as pd

class InertialDataset(Dataset):
    def __init__(self, source_file, d_model=8, src_seq_length=1, tgt_seq_length=1, split = True, train=True, validation=True, dtype=torch.float32, time_scale = 1):
        # Scale for time
        self.time_scale = time_scale
        # Number of features in input vector
        self.feature_num = 7
        # Minimal value saturation
        if d_model < self.feature_num:
            self.d_model = self.feature_num
        else:
            self.d_model = d_model

        if src_seq_length < 1:
            self.src_seq_length = 1
        else:
            self.src_seq_length = src_seq_length

        if tgt_seq_length < 1:
            self.tgt_seq_length = 1
        else:
            self.tgt_seq_length = tgt_seq_length

        # Datatype for torch tensors
        self.dtype = dtype

        # Reading data from csv, synchronizing MARG and MoCap
        df = pd.read_csv(source_file)

        time = [0]

        for i in range(1, df.shape[0]):
            time.append(time[-1] + df.iloc[i, -1])

        df = df.drop(['dt'], axis=1)
        data = df.to_numpy()
        time = np.array(time).reshape((data.shape[0], 1))
        data = np.hstack((time, data))

        # Optionally splitting the dataset into training and validation datasets
        # 1 dataset contains only 1 dataset type (Training, Validation or Test)
        d_partition = deepcopy(data)

        if split:
            if train:
                d_partition = d_partition[:49101, :]
            elif validation:
                d_partition = d_partition[49101:63211, :]
            else:
                d_partition = d_partition[63211:, :]

        # Creating vector for Timestamp only
        self.time = d_partition[:, 0]

        # Creating the matrix of labels
        self.Y = d_partition[:, -3:]
        # Inserting the required fields for model dimension match and for positional encoding purposes
        self.Y = self.const_cols_push_front(self.Y, self.d_model - self.Y.shape[1])

        # Creating the matrix of previous outputs
        self.prevY = deepcopy(self.Y)
        self.prevY[1:, :] = self.Y[0:-1, :]
        # Inserting the required fields for model dimension match and for positional encoding purposes
        self.prevY = self.const_cols_push_front(self.prevY, self.d_model - self.prevY.shape[1])

        # Creating the matrix of inputs
        self.X = d_partition[:, 1:-3]
        # Inserting the required fields for model dimension match and for positional encoding purposes
        self.X = self.const_cols_push_front(self.X, self.d_model - self.X.shape[1])

        # Calculating parameters for later standardization
        self.mean_src = self.X.mean(axis=0, dtype=np.float32)
        self.std_src = self.X.std(axis=0, dtype=np.float32)
        self.mean_tgt = self.prevY.mean(axis=0, dtype=np.float32)
        self.std_tgt = self.prevY.std(axis=0, dtype=np.float32)

    # Returns the producible number of sequences with required length from the dataset
    def __len__(self):
        return self.time.shape[0] - (self.src_seq_length-1)

    # Provides the sequence with required length from a starting index
    def get_sequence(self, data_stream, starting_index, seq_length):
        sequence = data_stream[starting_index:(starting_index + seq_length), ...]
        return sequence

    # Pushes the required number of 0 columns to the front of an existing numpy array
    def const_cols_push_front(self, np_arr, col_num):
        ret = np_arr
        if col_num > 0:
            const_cols = np.zeros((np_arr.shape[0], col_num))
            ret = np.hstack((const_cols, np_arr))

        return ret

    # Creates positional encoding for the input positions based on the model dimension too
    def get_pos_encoding(self, positions, d_model=8):
        # The angular frequency of trigonometric functions based on the publication "Attention Is All You Need"
        angular_freqs = 1 / (10000 ** (2 * np.arange(d_model // 2, dtype=float) / d_model))
        positions = np.reshape(positions, (np.size(positions), 1))
        # Input angles calculated for trigonometric functions
        angles = positions * angular_freqs
        pos_encoding = np.zeros((np.size(positions), d_model), dtype=float)
        # Sine on the even fields and cosine on the odd fields in each rows
        pos_encoding[:, 0::2] = np.sin(angles)
        pos_encoding[:, 1::2] = np.cos(angles)

        return pos_encoding

    def __getitem__(self, idx):
        # Converting the required encoder input sequence to torch tensor
        encoder_input = torch.tensor(self.get_sequence(self.X, idx, self.src_seq_length), dtype=self.dtype)
        # Converting the required label sequence to torch tensor
        label = torch.tensor(self.get_sequence(self.Y, idx, self.src_seq_length), dtype=self.dtype)
        # Getting timestamp sequence for position values
        positions = self.get_sequence(self.time, idx, self.src_seq_length)
        # Making the sequence positions to start from 0 and scaling based on positional encoding heat map experiment
        positions = (positions - positions[0])*self.time_scale
        # Calculating positional encoding and convering to torch tensor
        pos_encoding = self.get_pos_encoding(positions, self.d_model)
        pos_encoding = torch.tensor(pos_encoding, dtype=self.dtype)
        # Converting the required decoder input sequence to torch tensor
        decoder_input = torch.tensor(self.get_sequence(self.prevY, idx, self.tgt_seq_length), dtype=self.dtype)

        sample = {'encoder_input': encoder_input, 'pos_encoding': pos_encoding, 'decoder_input': decoder_input, 'label': label}
        return sample
