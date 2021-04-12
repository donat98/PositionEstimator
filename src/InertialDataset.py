import numpy as np
import pandas as pd
import math
from future.backports.datetime import time
import torch
from torch.utils.data import Dataset

class InertialDataset(Dataset):
    def __init__(self, source_file, d_model=8, seq_length=1, split = False, train=True, dtype=torch.float32, time_scale = (10**(-5))):
        # Scale for time
        self.time_scale = time_scale
        # Number of features in input vector
        self.feature_num = 7
        # Minimal value saturation
        if d_model < self.feature_num:
            self.d_model = self.feature_num
        else:
            self.d_model = d_model

        if seq_length < 1:
            self.seq_length = 1
        else:
            self.seq_length = seq_length

        self.dtype = dtype

        # Reading csv to pandas dataframe
        df = pd.read_csv(source_file)
        # Dropping the last outlier row
        df = df.drop([len(df) - 1])

        # Calculating the corresponding positions (X, Y, Z)
        df['POS_X'], df['POS_Y'], df['POS_Z'] = self.get_arc_pos(df['TIME2'].to_numpy(dtype=float), 515/1000/2, np.pi/2)

        # Creating a data frame for TIME2 only
        time_df = df['TIME2']

        # Selecting accelerations, quaternions and positions
        sensor_df = df.loc[:, 'ACC_X':'POS_Z']
        sensor_df = sensor_df.drop(['TIME1', 'TIME2'], axis=1)
        # Adding previous positions
        sensor_df['PREV_POS_X'], sensor_df['PREV_POS_Y'], sensor_df['PREV_POS_Z'] = sensor_df['POS_X'], sensor_df['POS_Y'], sensor_df['POS_Z']
        sensor_df.iloc[1:, -3], sensor_df.iloc[1:, -2], sensor_df.iloc[1:, -1] = sensor_df.iloc[0:-1, -6], sensor_df.iloc[0:-1, -5], sensor_df.iloc[0:-1, -4]

        # Creating the matrix of labels
        self.Y = sensor_df.loc[:, 'POS_X':'POS_Z'].to_numpy(dtype=float)
        # Inserting the required fields for model dimension match
        self.Y = self.const_cols_push_front(self.Y, self.d_model - self.Y.shape[1])

        # Creating the matrix of previous outputs
        self.prevY = sensor_df.loc[:, 'PREV_POS_X':'PREV_POS_Z'].to_numpy(dtype=float)
        # Inserting the required fields for positional encoding and model dimension match
        self.prevY = self.const_cols_push_front(self.prevY, self.d_model - self.prevY.shape[1])

        # Creating the matrix of inputs
        self.X = sensor_df.drop(['POS_X', 'POS_Y', 'POS_Z', 'PREV_POS_X', 'PREV_POS_Y', 'PREV_POS_Z'], axis=1).to_numpy(dtype=float)
        # Inserting the required fields for positional encoding
        self.X = self.const_cols_push_front(self.X, self.d_model - self.X.shape[1])

        # Convert time data frame to numpy array, correcting by offset and scaling
        self.time = time_df.to_numpy(dtype=float)

        self.mean_src = self.X.mean(axis=0, dtype=np.float32)
        self.std_src = self.X.std(axis=0, dtype=np.float32)
        self.mean_tgt = self.prevY.mean(axis=0, dtype=np.float32)
        self.std_tgt = self.prevY.std(axis=0, dtype=np.float32)

        if split:
            if train:
                self.X = self.X[:math.floor(len(self) * 0.7), :]
                self.Y = self.Y[:math.floor(len(self) * 0.7), :]
                self.prevY = self.prevY[:math.floor(len(self) * 0.7), :]
                self.time = self.time[:math.floor(len(self) * 0.7)]
            else:
                self.X = self.X[math.floor(len(self) * 0.7):, :]
                self.Y = self.Y[math.floor(len(self) * 0.7):, :]
                self.prevY = self.prevY[math.floor(len(self) * 0.7):, :]
                self.time = self.time[:math.floor(len(self) * 0.7)]

    def __len__(self):
        return self.time.shape[0] - (self.seq_length-1)

    def get_sequence(self, data_stream, starting_index, seq_length):
        sequence = data_stream[starting_index:(starting_index + seq_length), ...]
        return sequence

    def const_cols_push_front(self, np_arr, col_num):
        ret = np_arr
        if col_num > 0:
            const_cols = np.zeros((np_arr.shape[0], col_num))
            ret = np.hstack((const_cols, np_arr))

        return ret

    def const_rows_push_back(self, np_arr, row_num):
        ret = np_arr
        if row_num > 0:
            const_rows = np.zeros((row_num, np_arr.shape[-1]))
            ret = np.vstack((np_arr, const_rows))

        return ret

    def get_arc_pos(self, time, r, max_angle):
        t_norm = (time - time[0])/(time[-1] - time[0])
        pos_x = r * np.cos(max_angle * t_norm)
        pos_y = r * np.sin(max_angle * t_norm)
        pos_z = np.array([0] * np.size(t_norm))
        pos_z = pos_z.reshape(pos_y.shape)
        return pos_x, pos_y, pos_z

    def get_pos_encoding(self, positions, d_model=8):
        angular_freqs = 1 / (10000 ** (2 * np.arange(d_model // 2, dtype=float) / d_model))
        positions = np.reshape(positions, (np.size(positions), 1))
        angles = positions * angular_freqs
        pos_encoding = np.zeros((np.size(positions), d_model), dtype=float)
        pos_encoding[:, 0::2] = np.sin(angles)
        pos_encoding[:, 1::2] = np.cos(angles)

        return pos_encoding

    def __getitem__(self, idx):
        encoder_input = torch.tensor(self.get_sequence(self.X, idx, self.seq_length), dtype=self.dtype)
        label = torch.tensor(self.get_sequence(self.Y, idx, self.seq_length), dtype=self.dtype)
        positions = self.get_sequence(self.time, idx, self.seq_length)
        # Making the sequence positions to start from 0 and scaling based on positional encoding heat map experiment
        positions = (positions - positions[0])*self.time_scale
        pos_encoding = self.get_pos_encoding(positions, self.d_model)
        pos_encoding = torch.tensor(pos_encoding, dtype=self.dtype)
        decoder_input = torch.tensor(self.get_sequence(self.prevY, idx, self.seq_length), dtype=self.dtype)

        sample = {'encoder_input': encoder_input, 'pos_encoding': pos_encoding, 'decoder_input': decoder_input, 'label': label}
        return sample
