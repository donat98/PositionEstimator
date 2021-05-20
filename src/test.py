import matplotlib.pyplot as plt
from future.backports.datetime import time
from torch.utils.data import DataLoader
import InertialDataset as iDataset
import InertialTransformer as iTransformer

import torch
from torch import nn

if __name__ == 'test':

    # Dimension of a vector in the input and output sequences
    d_model = 8
    # Sequence length of encoder input
    src_seq_length = 8
    # Sequence length of decoder input
    tgt_seq_length = 8
    # Scale value for timestamps to creating proper positional encodings
    time_scale = 1e-5
    # Flag for using independent dataset for metrics or the training dataset
    independent = True

    # Selecting dataset
    if independent:
        test_ds = iDataset.InertialDataset('Robot_measurements/Combined2021-05-12_17-16-00.csv',
                                           d_model=d_model, src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length, split=True, train=False, validation=False, dtype=torch.float32,
                                           time_scale=time_scale)
    else:
        test_ds = iDataset.InertialDataset('Robot_measurements/Combined2021-05-12_17-16-00.csv',
                                           d_model=d_model, src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length, split=True, train=True, dtype=torch.float32,
                                           time_scale=time_scale)

    batch_size = 256

    # Defining data loader
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Saving the device name to use (cuda if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiating the transformer model
    transformer_model = iTransformer.InertialTransformer(d_model=d_model, nhead=2, num_encoder_layers=6,
                                                         num_decoder_layers=6, dim_feedforward=d_model * 4, dropout=0.1,
                                                         activation="relu",
                                                         mean_src=test_ds.mean_src, std_src=test_ds.std_src,
                                                         mean_tgt=test_ds.mean_tgt, std_tgt=test_ds.std_tgt).to(device)
    # Loading state of transformer model to test
    transformer_model.load_state_dict(torch.load('transformer_model.pth'))

    # Plotting the labels for position values
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(test_ds.Y[:, -3], test_ds.Y[:, -2], test_ds.Y[:, -1], c=test_ds.Y[:, -1], cmap='cool', s=1)
    cb = plt.colorbar(p, pad=0.2)
    ax.set(xlabel='X', ylabel='Y')
    plt.show()

    # Number of batches
    size = len(test_ds) // batch_size + 1
    # Variable for accumulating prediction error
    pred_error = 0

    # Previous prediction for the decoder input sequence
    prev_pred = test_ds[0]['decoder_input'][0].reshape((1, 1, d_model)).to(device)
    # Array for saving the predictions
    preds = prev_pred

    with torch.no_grad():
        for batch_num, sample in enumerate(test_dl):
            # Moving batch of labels to GPU
            label_b = sample['label'].to(device)

            # Moving batch of inputs to GPU
            encoder_in_b = sample['encoder_input'].to(device)
            pos_encoding_in_b = sample['pos_encoding'].to(device)

            # Iterating through batch
            for i in range(encoder_in_b.shape[0]):
                encoder_in = encoder_in_b[i, :, :].reshape((1, encoder_in_b.shape[1], encoder_in_b.shape[2]))
                pos_encoding_in = pos_encoding_in_b[i, :, :].reshape((1, pos_encoding_in_b.shape[1], pos_encoding_in_b.shape[2]))

                # Setting the previous prediction to decoder input
                decoder_in = prev_pred

                # Compute prediction
                pred = transformer_model(encoder_in, decoder_in, pos_encoding_in, src_seq_length)
                # Updating previous prediction
                prev_pred = pred
                # Saving prediction
                preds = torch.cat((preds, pred[:, -1, :].reshape(1, 1, d_model)), 1)

                # Accumulating the distance of predicted and dataset (label) points
                label = label_b[i, -1, -3:].reshape((1, 1, 3))
                diff = pred[:, -1, -3:].reshape((1, 1, 3)) - label
                last_distance = torch.norm(diff, 2)
                pred_error += last_distance


            if batch_num % 5 == 4:
                print(f'Prediction test state: {batch_num/size*100}% Done')
    print(f'Prediction test state: {100}% Done')

    # Calculating the average of distance errors
    pred_error /= len(test_ds)
    print(f'Average distance error: {pred_error}')
    print(f'Last distance error: {last_distance}')

    # Converting torch tensor to numpy arrays
    np_preds = preds.cpu().numpy().reshape((preds.shape[1], preds.shape[2]))
    pred_x = np_preds[:, -3]
    pred_y = np_preds[:, -2]
    pred_z = np_preds[:, -1]

    # Plotting predicted positions
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(pred_x , pred_y , pred_z , c=pred_z, cmap='autumn', s=1)
    cb = plt.colorbar(p, pad=0.2)
    ax.set(xlabel='X', ylabel='Y')
    plt.show()