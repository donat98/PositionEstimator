import numpy as np
import pandas as pd
import math
from future.backports.datetime import time
import torch
from torch.utils.data import Dataset

class InertialDataset(Dataset):
    def __init__(self, source_file, d_model=10, seq_length=1, train=True, dtype=torch.float64):
        # Minimal value saturation
        if d_model < 10:
            self.d_model = 10
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

        # Creating the matrix of inputs
        self.X = sensor_df.drop(['POS_X', 'POS_Y', 'POS_Z'], axis=1).to_numpy(dtype=float)
        # Inserting the required fields for positional encoding
        self.X = self.insert_const_cols(self.X, d_model - 10)

        # Convert time data frame to numpy array, correcting by offset and scaling
        self.time = time_df.to_numpy(dtype=float)

        if train:
            self.X = self.X[:math.floor(len(self) * 0.7), :]
            self.Y = self.Y[:math.floor(len(self) * 0.7), :]
            self.time = self.time[:math.floor(len(self) * 0.7)]
        else:
            self.X = self.X[math.floor(len(self) * 0.7):, :]
            self.Y = self.Y[math.floor(len(self) * 0.7):, :]
            self.time = self.time[:math.floor(len(self) * 0.7)]

    def __len__(self):
        return self.time.shape[0] - self.seq_length-1

    def get_sequence(self, data_stream, starting_index, seq_length):
        sequence = data_stream[starting_index:(starting_index + seq_length), ...]
        return sequence

    def insert_const_cols(self, np_arr, col_num):
        ret = np_arr
        if col_num > 0:
            const_cols = np.zeros((np_arr.shape[0], col_num))
            ret = np.hstack((const_cols, np_arr))

        return ret

    def get_arc_pos(self, time, r, max_angle):
        t_norm = (time - time[0])/(time[-1] - time[0])
        pos_x = r * np.cos(max_angle * t_norm)
        pos_y = r * np.sin(max_angle * t_norm)
        pos_z = np.array([0] * np.size(t_norm))
        pos_z = pos_z.reshape(pos_y.shape)
        return pos_x, pos_y, pos_z

    def __getitem__(self, idx):
        input_data = torch.tensor(self.get_sequence(self.X, idx, self.seq_length), dtype=self.dtype)
        label = torch.tensor(self.get_sequence(self.Y, idx, self.seq_length), dtype=self.dtype)
        positions = self.get_sequence(self.time, idx, self.seq_length)
        # Making the sequence positions to start from 0 and scaling based on positional encoding heat map experiment
        positions = torch.tensor((positions - positions[0])*(10**(-5)), dtype=self.dtype)

        sample = {'input_data': input_data, 'label': label, 'positions': positions}
        return sample