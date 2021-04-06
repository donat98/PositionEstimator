from time import time_ns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from future.backports.datetime import time
from torch.utils.data import DataLoader
import InertialDataset as ds
import torch

'''
def get_pos_encoding(positions, d_model = 8):
    angular_freqs = 1/(10000**(2*np.arange(d_model//2, dtype=float)/d_model))
    positions = np.reshape(positions, (np.size(positions), 1))
    angles = positions * angular_freqs
    pos_encoding = np.zeros((np.size(positions), d_model), dtype=float)
    pos_encoding[:, 0::2] = np.sin(angles)
    pos_encoding[:, 1::2] = np.cos(angles)

    return pos_encoding

def get_sequence(data_stream, starting_index, sec_length):
    sequence = data_stream[starting_index:(starting_index+sec_length), ...]
    return sequence

def insert_const_cols(np_arr, col_num):
    const_cols = np.zeros((np_arr.shape[0], col_num))
    return np.hstack((const_cols, np_arr))

def get_arc_pos(time, r, max_angle):
    t_norm = (time - time[0])/(time[-1] - time[0])
    pos_x = r * np.cos(max_angle * t_norm)
    pos_y = r * np.sin(max_angle * t_norm)
    pos_z = np.array([0] * np.size(t_norm))
    pos_z = pos_z.reshape(pos_y.shape)
    return pos_x, pos_y, pos_z
'''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    # Reading csv to pandas dataframe
    df = pd.read_csv('converted_saves-continous1/converted_save_continous1-4.csv')
    # Dropping the last outlier row
    df = df.drop([len(df)-1])

    # Adding arc positions
    df['POS_X'], df['POS_Y'], df['POS_Z'] = get_arc_pos(df['TIME2'].to_numpy(dtype=float), 515/2, np.pi)
    print(df)

    # Creating a data frame for TIME2 only
    time_df = df['TIME2']
    #print(time_df)
    # Selecting accelerations, quaternions and positions
    sensor_df = df.loc[:, 'ACC_X':'POS_Z']
    sensor_df = sensor_df.drop(['TIME1', 'TIME2'], axis=1)
    # Adding previous positions
    sensor_df['PREV_POS_X'], sensor_df['PREV_POS_Y'], sensor_df['PREV_POS_Z'] = sensor_df['POS_X'], sensor_df['POS_Y'], sensor_df['POS_Z']
    sensor_df.iloc[1:, -3], sensor_df.iloc[1:, -2], sensor_df.iloc[1:, -1] = sensor_df.iloc[0:-1, -6], sensor_df.iloc[0:-1, -5], sensor_df.iloc[0:-1, -4]
    print(sensor_df)

    # Creating the matrix of labels
    Y = sensor_df.loc[:, 'POS_X':'POS_Z'].to_numpy(dtype=float)
    print(Y)

    # Creating the matrix of inputs
    X = sensor_df.drop(['POS_X', 'POS_Y', 'POS_Z'], axis=1).to_numpy(dtype=float)
    X = insert_const_cols(X, 4)

    print(X.shape)

    # Calculating mean and standard deviation of inputs
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Convert time data frame to numpy array, correcting by offset and scaling
    time_np = time_df.to_numpy(dtype=float)
    sec_start = 60
    # Maximal sequence length is the number of input features extended to an even number
    sec_length = 10
    # Scale of time is based on position embedding heat map
    time_np_seq= get_sequence(time_np, sec_start, sec_length)
    time_np_seq = (time_np_seq-time_np_seq[0])*(10**(-5))
    print(time_np_seq)
    print(time_np_seq.shape)
    '''
    learning_ds = ds.InertialDataset('converted_saves-continous1/converted_save_continous1-4.csv', d_model=14, seq_length=1, train=True, dtype=torch.float64)
    learning_dl = DataLoader(learning_ds, batch_size=1, shuffle=True)

    for sample in learning_dl:
        print(f"Input data shape: {sample['input_data'].shape}")
        print(f"Label shape: {sample['label'].shape}")
        print(f"Positions shape: {sample['positions'].shape}")

    d_model = X.shape[1]
    pos_encoding = get_pos_encoding(time_np_seq, d_model)
    plt.pcolormesh(pos_encoding, cmap='hot')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    #plt.ylim((0, time_np_sec[-1]))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()