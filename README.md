# Next-slot OFDM-CSI Prediction
##### Mohamed Akrout, Faouzi Bellili, Amine Mezghani, Robert W. Heath
This repository contains the Python code of our paper: [Next-slot OFDM-CSI Prediction: Multi-head Self-attention or State Space Model?](https://arxiv.org/abs/2405.11072)

## Abstract
The ongoing fifth-generation (5G) standardization is exploring the use of deep learning (DL) methods to enhance the new radio (NR) interface. Both in academia and industry, researchers are investigating the performance and complexity of multiple DL architecture candidates for specific one-sided and two-sided use cases such as channel state estimation (CSI) feedback, CSI prediction, beam management, and positioning. In this paper, we set focus on the CSI prediction task and study the performance and generalization of the two main DL layers that are being extensively benchmarked within the DL community, namely, multi-head self-attention (MSA) and state-space model (SSM). We train and evaluate MSA and SSM layers to predict the next slot for uplink and downlink communication scenarios over urban microcell (UMi) and urban macrocell (UMa) OFDM 5G channel models. Our numerical results demonstrate that SSMs exhibit better prediction and generalization capabilities than MSAs only for SISO cases. For MIMO scenarios, however, the MSA layer outperforms the SSM one. While both layers represent potential DL architectures for future DL-enabled 5G use cases, the overall investigation of this paper favors MSAs over SSMs.

## Repository Structure
This repository contains the folder **layers**: it contains the code of the MSA and SSM layers. It also contains the following Python files:
  - `train.py` contains the code to train SSMs and MSAs.
  - `predict.py` contains the code to evaluate the OFDM-CSI prediction of the trained models.
  - `envs_channel.py` is an interface to the [Sionna](https://github.com/nvlabs/sionna) library simulating 5G OFDM environments.
    
## Running experiments
If this is your first time running the code, make sure to install all the required packages in `requirements.txt`

```bash
###### SISO example: uplink
# Run training
python train.py --model=ssm --scenario=siso --fc=5000000000 --snr_db=-30.0 --ue_speed=0.0 --channel=umi --tx_ant=1 --rx_ant=1
# Run prediction for siso
python predict.py --model=ssm --scenario=siso --fc=5000000000.0 --snr_db=-30.0 --ue_speed=0.0 --channel=umi --tx_ant=1 --rx_ant=1

###### MIMO example: downlink with 20 Tx antennas and 5 single-antenna users
# Run training
python train.py --model=ssm --scenario=mimo --fc=5000000000.0 --snr_db=-30.0 --ue_speed=0.0 --channel=umi --tx_ant=20 --rx_ant=5
# Run prediction
python predict.py --model=ssm --scenario=mimo --fc=5000000000.0 --snr_db=-30.0 --ue_speed=0.0 --channel=umi --tx_ant=20 --rx_ant=5
```

The prediction script evaluates the model on different scenarios by varying the SNR and the user speed, and saves the average MSE for each one of these scenarios in the csv file `prediction.csv`.

## Citing the paper (bib)

If you make use of our code, please make sure to cite our paper:
```
@misc{akrout2024nextslot,
      title={Next-slot OFDM-CSI Prediction: Multi-head Self-attention or State Space Model?}, 
      author={Mohamed Akrout and Faouzi Bellili and Amine Mezghani and Robert W. Heath},
      year={2024},
      eprint={2405.11072},
      archivePrefix={arXiv},
      primaryClass={cs.IT}
}
```
