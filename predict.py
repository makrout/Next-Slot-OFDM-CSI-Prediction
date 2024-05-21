from layers.mamba import MambaBlock
from layers.msa import MultiHeadAttention
from envs_channel import OfdmEnv, EnvConfig
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_performance(file_path, new_data):
    # define column names
    params_columns = ["fc", "SNR", "UE_speed", "channel", "Tx_ant", "Rx_ant"]
    train_params_columns = ["train_" + x for x in params_columns]
    test_params_columns = ["test_" + x for x in params_columns]
    columns = ["scenario", "model"] + train_params_columns + test_params_columns + ["MSE"]

    # add new data to the file if it exists or create a new one
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)

    # Append the new row to the existing content
    df.loc[len(df.index)] = new_data

    # update the new file
    df.to_csv(file_path, index=False)


def parse_args():
    '''
    Parse arguments from command line input
    '''
    parser = argparse.ArgumentParser(description='Simulation parameters')
    parser.add_argument('--scenario', type=str, help="wireless scenario", choices=['siso', 'mimo'], required=True)
    parser.add_argument('--ue_speed', type=str, help="UE speed", required=True)
    parser.add_argument('--snr_db', type=str, help="SNR", required=True)
    parser.add_argument('--channel', type=str, help="Channel type", choices=['umi', 'uma', 'rma'], required=True)
    parser.add_argument('--tx_ant', type=int, default="1", help="Number of transmit antennas", required=True)
    parser.add_argument('--rx_ant', type=int, default="1", help="Number of receive antennas", required=True)
    parser.add_argument('--fc', type=float, help="Carrier frequency", required=True)
    parser.add_argument('--model', type=str, help="Model type", choices=['ssm', 'msa'], required=True)
    parser.add_argument('--iterations', type=int, default="40", help="Test iterations")
    parser.add_argument('--batch_size', type=int, default="1", help="Batch size")
    parser.add_argument('--model_dir', type=str, default="models", help="Name of the folder containing the trained models")
    parser.add_argument('--out_dir', type=str, default="prediction.csv", help="Name of the output file")

    args, unknown = parser.parse_known_args()
    return args


class ModelArgs:
    def __init__(self, d_model):
        self.d_model = d_model  # D: number of channels of xt
        self.batch_size = 100  # B
        self.d_state = 7  # N: size of ht
        self.d_input = 7  # L: size of xt
        self.expand = 1  # E
        self.d_inner = int(self.expand * self.d_model)  # D
        self.dt_rank = math.ceil(self.d_model / 16)  # Delta
        self.d_conv = 4
        self.conv_bias = True
        self.bias = False

#################################
###   Sionna Data generator   ###
#################################


def get_data_sample(env, batch_size, snr_db, scenario):
    # generate a batch of channel
    _, _, channel = env(batch_size, snr_db, return_x=True)
    channel = torch.from_numpy(channel.numpy())
    if scenario == "mimo":
        channel = torch.squeeze(channel, (2, 3, 4))
        channel = torch.cat((channel.real, channel.imag), dim=3)
        channel = channel.view(batch_size, channel.shape[2], -1)
    elif scenario == "siso":
        channel = torch.squeeze(channel, (1, 2, 3, 4))
        channel = torch.cat((channel.real, channel.imag), dim=2)

    T = channel.shape[1]
    # generate a batch of training samples
    frame_width = 7

    for t in range(T - frame_width):
        t_end = t + frame_width

        x = channel[:, t:t_end, :]
        y = channel[:, t + 1:t_end + 1, :]

        yield x, y


if __name__ == '__main__':
    # define the range of test parameters
    FS = [5e9]
    CHANNELS = ['umi', 'uma', 'rma']
    TX_ANTS = [1]
    RX_ANTS = [1]
    UE_SPEEDS = [0.0, 10.0, 20.0, 30.0]
    TEST_SNRS = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    # parse experiment arguments
    args = parse_args()
    if args.snr_db == "all":
        sample_snr = True
    else:
        sample_snr = False
        try:
            args.snr_db = float(args.snr_db)
        except BaseException:
            Exception("`snr_db` must be a float or a string with value `all`")

    if args.ue_speed == "all":
        sample_ue_speed = True
    else:
        sample_ue_speed = False
        try:
            args.ue_speed = float(args.ue_speed)
        except BaseException:
            Exception("`ue_speed` must be a float or a string with value `all`")

    # define model path
    if args.scenario == 'siso':
        model_path = "{}/{}_{}_f_{}_Tx_{}_Rx_{}_{}_snr_{}_speed_{}".format(args.model_dir, args.scenario, args.channel, args.fc, args.tx_ant, args.rx_ant,
                                                                           args.model, args.snr_db, args.ue_speed)
    elif args.scenario == 'mimo':
        model_path = "{}/{}_{}_f_{}_Tx_{}_Rx_{}_nUE_{}_{}_snr_{}_speed_{}".format(args.model_dir, args.scenario, args.channel, args.fc, args.tx_ant, 1,
                                                                                  args.rx_ant, args.model, args.snr_db, args.ue_speed)

    # define the models' parameters
    if args.scenario == 'siso':
        model_args = ModelArgs(d_model=144)
    elif args.scenario == 'mimo':
        model_args = ModelArgs(d_model=1000)
    # initialize the model
    if args.model == "ssm":
        # create the mamba layer
        layer = MambaBlock(model_args)
    elif args.model == "msa":
        # create the multi-head attention layer
        # with two heads for real and imaginary parts
        layer = MultiHeadAttention(model_args.d_model, 2)

    # load the model parameters
    layer.load_state_dict(torch.load(model_path))
    layer.eval()

    # Inference part
    losses = []
    times = []

    for fc in FS:
        for tx_ant in TX_ANTS:
            for rx_ant in RX_ANTS:
                for channel in CHANNELS:
                    for ue_speed in UE_SPEEDS:
                        for snr_db in TEST_SNRS:
                            print("Predicting at snr = {} and ue_speed = {}".format(snr_db, ue_speed))
                            # in-distribution and out-distribution test loop
                            pbar = tqdm(range(args.iterations))
                            for i in pbar:
                                # set the Sionna data generator with the appropriate test parameters
                                if args.scenario == 'siso':
                                    config = {'carrier_frequency': args.fc, 'num_rx_antennas': args.rx_ant, 'n_ues': args.tx_ant, 'ue_speed': args.ue_speed,
                                              'scenario': args.channel, 'direction': "uplink"}
                                elif args.scenario == 'mimo':
                                    config = {'carrier_frequency': args.fc, 'num_rx_antennas': 1, 'n_ues': args.rx_ant, 'num_streams_per_tx': args.tx_ant,
                                              'ue_speed': args.ue_speed, 'scenario': args.channel, 'pilot_pattern': "kronecker", 'fft_size': args.tx_ant * args.rx_ant,
                                              'direction': "downlink"}

                                env_config = EnvConfig()
                                env_config.from_dict(config)
                                env = OfdmEnv(env_config)

                                # test loop
                                for (x, y) in get_data_sample(env, args.batch_size, snr_db, scenario=args.scenario):
                                    start_time = time.time()

                                    if args.model == "ssm":
                                        y_pred = layer(x)
                                    elif args.model == "msa":
                                        y_pred = layer(x, x, x)
                                    inference_time = time.time() - start_time
                                    loss = F.mse_loss(y_pred, y)
                                    losses.append(loss.item())
                                    times.append(inference_time)
                                    pbar.set_description("MSE = {}".format(loss))

                            # print performance
                            mean_mse = np.mean(losses)
                            mean_inference_time = np.sum(inference_time) / (7 * args.iterations * args.batch_size) # 7 because the slot has 7 OFDM symbols
                            print("Average mse = {}, std = {}\n".format(mean_mse, np.std(losses)))

                            # save the prediction performance
                            train_params = [args.fc, str(args.snr_db), args.ue_speed, str(args.channel), args.tx_ant, args.rx_ant]
                            test_params = [fc, snr_db, ue_speed, str(channel), tx_ant, rx_ant]

                            data = [str(args.scenario), str(args.model)] + train_params + test_params + [mean_mse]
                            save_performance(args.out_dir, data)
