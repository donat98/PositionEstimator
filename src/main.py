from time import time_ns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from future.backports.datetime import time
from torch.utils.data import DataLoader
import InertialDataset as iDataset
import InertialTransformer as iTransformer

import torch
from torch import nn

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


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

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
    d_model = 8
    seq_length = 5
    time_scale = (10**(-5))
    learning_ds = iDataset.InertialDataset('converted_saves-continous1/converted_save_continous1-4.csv', d_model=d_model, seq_length=seq_length, split=False, dtype=torch.float32, time_scale=time_scale)
    validation_ds = iDataset.InertialDataset('converted_saves-continous1/converted_save_continous1-8.csv', d_model=d_model, seq_length=seq_length, split=False, dtype=torch.float32, time_scale=time_scale)

    '''
    for sample in learning_ds:
        
        sample['input_data'] = torch.transpose(sample['input_data'], 0, 1)
        sample['label'] = torch.transpose(sample['label'], 0, 1)
        sample['positions'] = torch.transpose(sample['positions'], 0, 1)

        print(f"Encoder input shape: {sample['encoder_input'].shape}")
        print(f"Positional encoding shape: {sample['pos_encoding'].shape}")
        print(f"Decoder input shape: {sample['decoder_input'].shape}")
        print(f"Label shape: {sample['label'].shape}")

    print(f"{generate_square_subsequent_mask(10).shape}")
    '''


    #Positional encoding example heat map
    pos_encoding = learning_ds[0]['pos_encoding'].numpy()
    plt.pcolormesh(pos_encoding, cmap='hot')
    plt.xlabel('Depth')
    plt.xlim((0, learning_ds.d_model))
    #plt.ylim((0, time_np_sec[-1]))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()

    batch_size = 10
    learning_rate = 10**(-3)
    epochs = 50 #50

    learning_dl = DataLoader(learning_ds, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_model = iTransformer.InertialTransformer(d_model=d_model, nhead=2, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=d_model*4, dropout=0.1, activation="relu",
                                                         mean_src=learning_ds.mean_src, std_src=learning_ds.std_src, mean_tgt=learning_ds.mean_tgt, std_tgt=learning_ds.std_tgt).to(device)

    '''
    for sample in learning_dl:
        sample_raw = sample['encoder_input']
        sample_standardized = transformer_model.standardize(sample['encoder_input'], torch.from_numpy(np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float32)), torch.from_numpy(np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)))
    '''

    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=10e-9)
    loss_fn = nn.MSELoss()

    #Lists for result plotting
    batch_num_train = []
    batch_num_valid = []
    training_loss_hist = []
    valid_loss_hist = []
    est_valid_loss_hist = []

    for epoch in range(epochs):
        #Number of batches in an epoch
        size = len(validation_ds) // batch_size + 1

        for batch_num, sample in enumerate(learning_dl):
            #Moving batch of label to GPU
            label = sample['label'].to(device)

            #Apply positional encoding and moving the results to GPU
            encoder_in = (sample['encoder_input'] + sample['pos_encoding']).to(device)
            decoder_in = (sample['decoder_input'] + sample['pos_encoding']).to(device)

            # Compute prediction and loss
            pred = transformer_model(encoder_in, decoder_in, seq_length, seq_length)
            training_loss = loss_fn(pred, label)

            # Backpropagation
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            # Save result for plot
            batch_num_train.append(epoch*size + batch_num)
            training_loss_hist.append(training_loss.item())

            #Indexing starts from 0 and every 5th training loss is printed
            if batch_num % 5 == 4:
                print(f'Loss: {training_loss}')

        #Variables for validation on an epoch
        validation_loss = 0
        est_validation_loss = 0

        with torch.no_grad():
            for sample in validation_dl:
                # Moving batch of label to GPU
                label = sample['label'].to(device)

                # Apply positional encoding and moving the results to GPU
                encoder_in = (sample['encoder_input'] + sample['pos_encoding']).to(device)
                decoder_in = (sample['decoder_input'] + sample['pos_encoding']).to(device)

                # Compute prediction and loss
                pred = transformer_model(encoder_in, decoder_in, seq_length, seq_length)
                validation_loss += loss_fn(pred, label)
                est_validation_loss += loss_fn(pred[:, -1, :], label[:, -1, :])

        # Average validation loss calculation for epoch
        validation_loss /= size
        est_validation_loss /= size

        # Save results for plot
        batch_num_valid.append((epoch + 1)*size)
        valid_loss_hist.append(validation_loss.item())
        est_valid_loss_hist.append(est_validation_loss.item())

        print(f"Avg validation loss: {validation_loss}")
        print(f"Avg validation loss for estimated position: {est_validation_loss}")

    plt.plot(batch_num_train, training_loss_hist)
    plt.plot(batch_num_valid, valid_loss_hist)
    plt.plot(batch_num_valid, est_valid_loss_hist)
    plt.xlabel('Batch number')
    plt.xlim((0, batch_num_train[-1]))
    plt.ylim((0, training_loss_hist[0]))
    plt.ylabel('Loss')
    plt.title("Training results")
    plt.legend(["Training loss", "Validation loss", "Validation loss for estimation"])
    plt.show()
