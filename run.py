from argparse import ArgumentParser
from inspect import isfunction
from os import path as osp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataloading import *
from model.PopularityPredictor import TimeRNNPopularityPredictor, TimeGNNPopularityPredictor, BasePopularityPredictor
from nn.readout import TimeMultiAttendReadout


def main():
    parser = ArgumentParser(prog='Popularity Predictor')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TimeRNNPopularityPredictor.add_model_specific_args(parser)
    parser = BaseDataModule.add_model_specific_args(parser)
    parser = TimeMultiAttendReadout.add_model_specific_args(parser)

    parser.add_argument('--dataloader', type=str, choices=['FOREST', 'weibo'], default='FOREST',
                        help='Type of dataloader, to load data in different format (default %(default)s).')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience of early stopping (default %(default)s).')
    parser.add_argument('--model', type=str, choices=['TimeRGNN', 'TimeGNN', 'UserGNN'], default='TimeRGNN',
                        help='The type of model (default $(default)s).'
                             'TimeRGNN (inherited from TimeGNN): combine heterogeneous GCN and time RNN;'
                             'TimeGNN (inherited from UserGNN): adding time nodes to graph for heterogeneous GCN;'
                             'UserGNN: base heterogeneous GCN on initial user nodes.'
                             )
    args = parser.parse_args()
    dict_args = vars(args)
    non_func_args = {k: v for k, v in filter(lambda x: not isfunction(x[1]), dict_args.items())}
    print('args:\n', non_func_args)
    arg_dataloader = args.dataloader

    patience = dict_args['patience']  # arg

    if arg_dataloader == 'weibo':
        assert dict_args['data_name'] in ['repost', 'topic', 'twitter'], \
            f'dataset {dict_args["data_name"]} from {arg_dataloader} not exists'
        datamodule = WeiboTopicDataModule.from_argparse_args(args, on_multi_files=False)

    elif arg_dataloader == 'FOREST':
        assert dict_args['data_name'] in ['twitter', 'douban'], \
            f'dataset {dict_args["data_name"]} from {arg_dataloader} not exists'
        datamodule = MarcoTaskDataModule.from_argparse_args(args)

    # init model
    etypes = [e[1] for e in datamodule.etypes]
    dataset_info = datamodule.dataset.filename
    logger = TensorBoardLogger(osp.join("lightning_logs", dataset_info, f'batch{dict_args["batch_size"]}'),
                               name=args.model, default_hp_metric=False)
    early_stopping = EarlyStopping(monitor='valid loss', patience=patience, verbose=True, mode='min',
                                   # min_delta=1e-4,
                                   # stopping_threshold=1e-3,
                                   )
    # To detect whether ReLU caused the death of all neurons.
    # ev_early_stopping = EarlyStopping(monitor='valid percentage error', patience=patience, verbose=True, mode='min',
    #                                   divergence_threshold=1
    #                                   )
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[early_stopping,])

    if args.model == 'TimeRGNN':
        readout = TimeMultiAttendReadout.from_argparse_args(args)
        model = TimeRNNPopularityPredictor.from_argparse_args(args, readout=readout, rel_names=etypes,
                                                              require_process=False)
    elif args.model == 'TimeGNN':
        readout = TimeMultiAttendReadout.from_argparse_args(args)
        model = TimeGNNPopularityPredictor.from_argparse_args(args, readout=readout, rel_names=etypes,
                                                              require_process=False)
    elif args.model == 'UserGNN':
        args.readout_use = 'user'
        readout = TimeMultiAttendReadout.from_argparse_args(args)
        model = BasePopularityPredictor.from_argparse_args(args, readout=readout, rel_names=etypes,
                                                           require_process=False)
    if args.gpus:
        model = model.to(device='cuda')

    if arg_dataloader == 'weibo':
        if datamodule.on_multi_files:
            # if not datamodule.dataset.has_processed:
            datamodule.process(model.process_graph, save=True)
            model.require_process = False
        else:
            model.require_process = True
    elif arg_dataloader == 'FOREST':
        datamodule.setup('test')
        datamodule.setup('fit')
        datamodule.train_dataset.process_each(model.process_graph)
        datamodule.valid_dataset.process_each(model.process_graph)
        datamodule.test_dataset.process_each(model.process_graph)
        model.require_process = False

    # 继续某一次学习
    ckpt = "lightning_logs/version_2H/checkpoints/epoch=10.ckpt"
    # model = TimeRNNPopularityPredictor.load_from_checkpoint(ckpt, rel_names=etypes, datamodule=datamodule)

    model.datamodule = datamodule

    # find better learning rate
    trainer.tune(model, datamodule=model.datamodule, lr_find_kwargs={'early_stop_threshold': None,
                                                                           'min_lr': 1e-6,
                                                                           'max_lr': 5e-3
                                                                           })
    # lr_finder = rst['lr_find']
    # lr_finder.results
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # new_lr = lr_finder.suggestion()

    # train
    trainer.fit(model, datamodule=model.datamodule)

    # test
    trainer.test(model, datamodule=model.datamodule)


if __name__ == '__main__':
    main()
