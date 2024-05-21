from layers.mamba import MambaBlock
from layers.msa import MultiHeadAttention
from envs_channel import OfdmEnv, EnvConfig
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
import math
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    parser.add_argument('--n_epochs', type=int, default="5", help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default="100", help="Batch size")
    parser.add_argument('--learning_rate', type=float, default="1e-3", help="Learning rate")
    parser.add_argument('--out_dir', type=str, default="models", help="Name of the folder where trained models are saved")
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
    # define the range of SNR and user speed
    UE_SPEEDS = [0.0, 10.0, 20.0, 30.0]
    SNRS = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    # Parse experiment arguments
    args = parse_args()

    # create the models folder to save trained networks
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

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

    # Define the models' parameters
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

    # Training part
    optim = torch.optim.RMSprop(layer.parameters(), lr=args.learning_rate)

    layer.train()
    pbar = tqdm(range(args.n_epochs))
    for i in pbar:

        if sample_snr:
            args.snr_db = random.sample(SNRS, 1)[0]
        if sample_ue_speed:
            args.ue_speed = args.ue_speed = random.sample(UE_SPEEDS, 1)[0]

        # set the Sionna data generator with the appropriate parameters
        # for siso uplink or MIMO downlink scenarios
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

        # Training loop
        for (x, y) in get_data_sample(env, args.batch_size, args.snr_db, scenario=args.scenario):
            if args.model == "ssm":
                y_pred = layer(x)
            elif args.model == "msa":
                y_pred = layer(x, x, x)

            loss = F.mse_loss(y_pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # update the tqdm progress bar
        pbar.set_description("MSE = {}".format(loss))

    # save the model
    if sample_snr:
        args.snr_db = "all"
    if sample_ue_speed:
        args.ue_speed = "all"
    if args.scenario == 'siso':
        model_path = "{}/{}_{}_f_{}_Tx_{}_Rx_{}_{}_snr_{}_speed_{}".format(args.out_dir, args.scenario, args.channel, args.fc, args.tx_ant, args.rx_ant,
                                                                           args.model, args.snr_db, args.ue_speed)
    elif args.scenario == 'mimo':
        model_path = "{}/{}_{}_f_{}_Tx_{}_Rx_{}_nUE_{}_{}_snr_{}_speed_{}".format(args.out_dir, args.scenario, args.channel, args.fc, args.tx_ant, 1,
                                                                                  args.rx_ant, args.model, args.snr_db, args.ue_speed)
    torch.save(layer.state_dict(), model_path)
