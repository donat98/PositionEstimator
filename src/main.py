from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import InertialDataset as iDataset
import InertialTransformer as iTransformer

import torch
from torch import nn

if __name__ == '__main__':

    # Dimension of a vector in the input and output sequences
    d_model = 8
    # Sequence length of encoder input
    src_seq_length = 8
    # Sequence length of decoder input
    tgt_seq_length = 8
    # Scale value for timestamps to creating proper positional encodings
    time_scale = 1e-5

    # Creating a training dataset
    learning_ds = iDataset.InertialDataset('Robot_measurements/Combined2021-05-12_17-16-00.csv',
                                           d_model=d_model, src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length, split=True, train=True, dtype=torch.float32,
                                           time_scale=time_scale)
    # Creating a validation dataset
    validation_ds = iDataset.InertialDataset('Robot_measurements/Combined2021-05-12_17-16-00.csv',
                                             d_model=d_model, src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length, split=True, train=False, validation=True, dtype=torch.float32,
                                             time_scale=time_scale)

    # Creating a test dataset
    test_ds = iDataset.InertialDataset('Robot_measurements/Combined2021-05-12_17-16-00.csv',
                                            d_model=d_model, src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length, split=True, train=False, validation=False, dtype=torch.float32,
                                             time_scale=time_scale)

    # Positional encoding example heat map
    pos_encoding = learning_ds[0]['pos_encoding'].numpy()
    plt.pcolormesh(pos_encoding, cmap='hot')
    plt.xlabel('Depth')
    plt.xlim((0, learning_ds.d_model))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()

    # Plotting the labels for position values
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(learning_ds.Y[:, -3], learning_ds.Y[:, -2], learning_ds.Y[:, -1], c=learning_ds.Y[:, -1], cmap='cool', s=1)
    cb = plt.colorbar(p, pad=0.2)
    ax.set(xlabel='X', ylabel='Y')
    plt.show()

    batch_size = 256
    # The initial lerarning rate, this number will be multiplied by the scheduler (actual_learning_rate = learning_rate * lr_lambda(step_num))
    learning_rate = 1
    # Flag to load a saved model and continue training or begin a new training process
    cont_saved = False
    epochs = 25  # 25
    # Number of learning steps while learning rate is increased
    warmup_steps = 500
    # Starting index of output feature slice for loss calculation (Recommended: 0 or -3)
    error_starting_idx = 0

    # Defining the data loaders
    learning_dl = DataLoader(learning_ds, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Saving the device name to use (cuda if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiating the transformer model
    transformer_model = iTransformer.InertialTransformer(d_model=d_model, nhead=2, num_encoder_layers=6,
                                                         num_decoder_layers=6, dim_feedforward=d_model * 4, dropout=0.1,
                                                         activation="relu",
                                                         mean_src=learning_ds.mean_src, std_src=learning_ds.std_src,
                                                         mean_tgt=learning_ds.mean_tgt, std_tgt=learning_ds.std_tgt).to(device)

    # Optional saved state loading
    if cont_saved:
        transformer_model.load_state_dict(torch.load('transformer_model.pth'))

    # Setting the optimizer based on the publication "Attention Is All You Need"
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=10e-9)
    loss_fn = nn.MSELoss()

    # Lists for result plotting
    batch_num_train = []
    batch_num_valid = []
    training_loss_hist = []
    valid_loss_hist = []
    est_valid_loss_hist = []
    # Big initial value for best validation loss
    best_loss = 999999999

    # Function and scheduler for calculating and setting adaptive learning rate
    lr_lambda = lambda step_num: 128*min(float((step_num+1))**(-1.5), float((step_num+1))*warmup_steps**(-2.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=True)
    #Getting the first calculated learning rate
    scheduler.step()

    # Number of batches in an epoch
    size = len(learning_ds) // batch_size + 1

    for epoch in range(epochs):

        for batch_num, sample in enumerate(learning_dl):
            # Moving batch of labels to GPU (if available)
            label = sample['label'].to(device)

            # Moving the encoder and decoder inputs to GPU (if available)
            encoder_in = sample['encoder_input'].to(device)
            decoder_in = sample['decoder_input'].to(device)

            # Moving the positional encodings to GPU (if available)
            pos_encoding_in = sample['pos_encoding'].to(device)

            # Compute prediction and loss
            pred = transformer_model(encoder_in, decoder_in, pos_encoding_in, src_seq_length)
            training_loss = loss_fn(pred[:, :, error_starting_idx:], label[:, :, error_starting_idx:])

            # Backpropagation
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            scheduler.step()

            # Save result for plot
            batch_num_train.append(epoch * size + batch_num)
            training_loss_hist.append(training_loss.item())

            # Indexing starts from 0 and every 5th training loss is printed
            if batch_num % 5 == 4:
                print(f'Loss: {training_loss}')

        # Variables for validation on an epoch
        validation_loss = 0
        est_validation_loss = 0

        with torch.no_grad():
            for sample in validation_dl:
                # Moving batch of labels to GPU (if available)
                label = sample['label'].to(device)

                # Moving the encoder and decoder inputs to GPU (if available)
                encoder_in = sample['encoder_input'].to(device)
                decoder_in = sample['decoder_input'].to(device)

                # Moving the positional encodings to GPU (if available)
                pos_encoding_in = sample['pos_encoding'].to(device)

                # Compute prediction and loss
                pred = transformer_model(encoder_in, decoder_in, pos_encoding_in, src_seq_length)
                validation_loss += loss_fn(pred[:, :, error_starting_idx:], label[:, :, error_starting_idx:])
                # Loss for the concrete X, Y, Z coordinates
                est_validation_loss += loss_fn(pred[:, -1, -3:], label[:, -1, -3:])

        # Average validation loss calculation for epoch
        validation_loss /= size
        est_validation_loss /= size

        # Saving the state of current model if the validation loss is less than the best one
        if validation_loss < best_loss:
            # Updating best loss
            best_loss = validation_loss
            torch.save(transformer_model.state_dict(), 'transformer_model.pth')

        # Save results for plot
        batch_num_valid.append((epoch + 1) * size)
        valid_loss_hist.append(validation_loss.item())
        est_valid_loss_hist.append(est_validation_loss.item())

        print(f"Avg validation loss: {validation_loss}")
        print(f"Avg validation loss for estimated position: {est_validation_loss}")
        print(f"{(epoch+1)/epochs*100}% Done")

    # Plotting learning curves
    plt.plot(batch_num_train, training_loss_hist)
    plt.plot(batch_num_valid, valid_loss_hist)
    plt.plot(batch_num_valid, est_valid_loss_hist)
    plt.xlabel('Batch number')
    plt.xlim((0, batch_num_train[-1]))
    plt.ylim((0, training_loss_hist[0]))
    plt.ylabel('Loss')
    plt.title("Training results")
    plt.legend(["Training loss", "Validation loss", "Validation loss for last element"])
    plt.show()

    # Calculating loss on test dataset

    size = len(test_ds) // batch_size + 1
    test_loss = 0

    with torch.no_grad():
        for batch_num, sample in enumerate(test_dl):
            # Moving batch of labels to GPU (if available)
            label = sample['label'].to(device)

            # Moving the encoder and decoder inputs to GPU (if available)
            encoder_in = sample['encoder_input'].to(device)
            decoder_in = sample['decoder_input'].to(device)

            # Moving the positional encodings to GPU (if available)
            pos_encoding_in = sample['pos_encoding'].to(device)

            # Compute prediction and loss
            pred = transformer_model(encoder_in, decoder_in, pos_encoding_in, src_seq_length)
            test_loss += loss_fn(pred[:, :, error_starting_idx:], label[:, :, error_starting_idx:])

            if batch_num % 5 == 4:
                print(f'Loss test state: {batch_num / size * 100}% Done')

    print(f'Loss test state: {100}% Done')

    # Average test loss calculation
    test_loss /= size
    print(f"Avg test loss: {test_loss}")

    # Prediction test

    # Number of batches
    size = len(learning_ds) // batch_size + 1

    # Previous prediction for the decoder input sequence
    prev_pred = learning_ds[0]['decoder_input'][0].reshape((1, 1, d_model)).to(device)
    # Array for saving the predictions
    preds = prev_pred

    with torch.no_grad():
        for batch_num, sample in enumerate(learning_dl):
            # Moving batch of inputs to GPU
            encoder_in_b = sample['encoder_input'].to(device)
            pos_encoding_in_b = sample['pos_encoding'].to(device)

            # Iterating through batch
            for i in range(encoder_in_b.shape[0]):
                encoder_in = encoder_in_b[i, :, :].reshape((1, encoder_in_b.shape[1], encoder_in_b.shape[2]))
                pos_encoding_in = pos_encoding_in_b[i, :, :].reshape(
                    (1, pos_encoding_in_b.shape[1], pos_encoding_in_b.shape[2]))

                # Setting the previous prediction to decoder input
                decoder_in = prev_pred

                # Computing prediction
                pred = transformer_model(encoder_in, decoder_in, pos_encoding_in, src_seq_length)
                # Updating previous prediction
                prev_pred = pred
                # Saving prediction
                preds = torch.cat((preds, pred[:, -1, :].reshape(1, 1, d_model)), 1)

            if batch_num % 5 == 4:
                print(f'Prediction test state: {batch_num / size * 100}% Done')
    print(f'Prediction test state: {100}% Done')

    # Converting torch tensor to numpy arrays
    np_preds = preds.cpu().numpy().reshape((preds.shape[1], preds.shape[2]))
    pred_x = np_preds[:, -3]
    pred_y = np_preds[:, -2]
    pred_z = np_preds[:, -1]

    # Plotting predicted positions
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(pred_x, pred_y, pred_z, c=pred_z, cmap='autumn', s=1)
    cb = plt.colorbar(p, pad=0.2)
    ax.set(xlabel='X', ylabel='Y')
    plt.show()

