# HERI-GCN

## Description

Our HERI-GCN is implemented mainly based on the following libraries (see the README file in source code folder for more details):

- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://www.pytorchlightning.ai/
- DGL: https://www.dgl.ai/
- PytorchNLP: https://github.com/PetrochukM/PyTorch-NLP
- Networkx: https://networkx.org/

### File Tree

Project file structure and description:

```python
HERI-GCN
├─ README.md
├─ __init__.py
├─ dataloading	# package of dataloading
│    ├─ __init__.py
│    ├─ datamodule.py	
│    └─ dataset	# package of dataset class
│           ├─ WeiboTopicDataset.py	# process data of csv format 
│           └─ __init__.py
├─ hooks.py	# hooks of model
├─ model	# package of models (HERI-GCN and its variants)
│    ├─ PopularityPredictor.py
│    ├─ __init__.py
├─ nn	# package of neural network layers
│    ├─ __init__.py
│    ├─ conv.py	# graph convolution layer
│    └─ readout.py	# output layer
├─ requirements.txt	
├─ run.py	# running entrance
└─ utils	# utils of drawing, dataloading, tensor and parameter propcessing
       ├─ __init__.py
       ├─ arg_parse.py
       ├─ dataloading.py
       ├─ drawing.py # heterogenous graph drawing
       ├─ output.py
       └─ utils.py
```

### Models

In `model` package, we designed three models:

|                     Model                     |                       Input Relations                        |              Computation               |
| :-------------------------------------------: | :----------------------------------------------------------: | :------------------------------------: |
|  `BasePopularityPredictor`<br>(HERI-GCN-UG)   |         (user, repost, user)<br>(user, follow, user)         |           Heterogeneous  GCN           |
| `TimeGNNPopularityPredictor`<br>(HERI-GCN-TG) | (user, repost, user)<br/>(user, follow, user)<br/>(user, post at, time)<br/>(time, contain, user)<br/>(time, past to, time) |           Heterogeneous  GCN           |
|  `TimeRNNPopularityPredictor`<br>(HERI-GCN)   | (user, repost, user)<br/>(user, follow, user)<br>(user, post at, time)<br/>(time, contain, user)<br>(time, past to, time) | Heterogeneous  GCN <br>Integrated  RNN |

`BasePopularityPredictor` is the basic predictor model,  `TimeGNNPopularityPredictor` inherit from `BasePopularityPredictor`, and `TimeRNNPopularityPredictor` inherit from `TimeGNNPopularityPredictor`.

## Installation

Installation requirements are described in `requirements.txt`.

- Use pip:

  ```bash
  pip install -r requirements.txt
  ```

- Use anaconda:

  ```bash
  conda install --file requirements.txt
  ```

## Usage

Get helps for all parameters of data processing, training and optimization:

```bash
python run.py  --help
```

Run:

```bash
Python run.py --paramater value
```

### Experiment Settings

The flag-parameter is boolean type, it works with default value, and the specific value is not necessary.

The settings of the ==highlighting== parameters are analyzed in detail in the following section.

**Basic settings** (common for all experiments, and recommend to use `auto_lr_find` to optimize the learning rate):

|      Parameter       |       Value       | Description                                                               |
| :------------------: | :---------------: | ------------------------------------------------------------------------- |
|     readout_use      |        all        | Use both time feature and user feature to output, or just one of them.    |
|       in_feats       |        16         | Input feature dimension.                                                  |
|      hid_feats       |        32         | Hidden feature dimension.                                                 |
|      out_feats       |        64         | Output feature dimension.                                                 |
|     dropout_rate     |        0.3        | Dropout rate.                                                             |
|      rnn_feats       |        32         | RNN feature dimension.                                                    |
|      batch_size      | 4 (8 for Twitter) | Batch size.                                                               |
|    learning_rate     |       5e-3        | Learning rate.                                                            |
|     weight_decay     |       5e-3        | Weight of L2 regulation.                                                  |
|      gcn_layers      |         3         | Number of heterogeneous GCN layers.                                       |
|      rnn_layers      |         2         | Number of RNN layers.                                                     |
|  rnn_bidirectional   |    True (flag)    | Is RNN bi-directional.                                                    |
|         rnn          |        gru        | Instance of RNN module (GRU or LSTM).                                     |
|      time_nodes      |        50         | Number of time nodes added into heterogeneous graph.                      |
|        split         |       time        | Split user sequence into time intervals according to user number or time. |
|         hop          |         1         | Hops of follower sampling.                                                |
|       patience       |        20         | Patience of training early stopping.                                      |
| readout_weighted_sum |    True (flag)    | Use weighted sum or product of user feature and time feature to output.   |

**Special settings**:

|     Parameter      |                Default value                | Description                                                  |
| :----------------: | :-----------------------------------------: | ------------------------------------------------------------ |
|       model        |                  TimeRGNN                   | Specify the model, choice from [UserGNN, TimeGNN, TimeRGNN]. |
|     data_name      |                    topic                    | Specify the dataset, choice from [twitter, repost, topic].   |
|    time_window     |                     24                      | Specify the time window to observe (hours).                  |
|     dataloader     |                    weibo                    | Specify dataloader for a certain data format, the default dataloader loads data from csv format. |
|      raw_dir       |                   ./data                    | Specify the data directory.                                  |
| min_cascade_length | 20 for weibo dataset, 5 for twitter dataset | Minimal cascade length, the shorter cascades will be filter out in data processing. |

**Other optimization settings** (inherited from [pytorch_lightning.Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags)):

|       Parameter       | Recommend value                                                     | Description                                                                                                                             |
| :-------------------: | :------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
|         gpus          | 0 for CPU only, 1 for single GPU.                                   | Number of GPUs to train on (int), or which GPUs to train on (list:[int, str]).                                                          |
|     num_processes     | 0 for Windows, other systems can be set up according to your needs. | Number of processes to train with.                                                                                                      |
|      max_epochs       | 100                                                                 | Stop training once this number of epochs is reached.                                                                                    |
|     auto_lr_find      | True (flag)                                                         | Runs a learning rate finder algorithm to find optimal initial learning rate.                                                            |
| auto_scale_batch_size | True (flag), power, binsearch                                       | Automatically tries to find the largest batch size that fits into memory, before any training.                                          |
|   gradient_clip_val   | 0.005                                                               | Gradient clipping value.                                                                                                                |
| stochastic_weight_avg | True (flag)                                                         | Stochastic Weight Averaging (SWA), to smooths the loss landscape thus making it harder to end up in a local minimum during optimization.|

## Cite Us

```bib
@inproceedings{HERIGCN_2022,
title = {},
author = {}
year = {}
pages = {}
}
```





 