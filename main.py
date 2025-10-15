import os
from datetime import datetime
import argparse
import numpy as np
import torch

from models.handler_grud import (
    train,
    test,
    load_PeMS07,
    load_elec,
    load_MAPSS,
    load_solar,
    split_data,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Flow control
    p.add_argument('--train', type=bool, default=True)
    p.add_argument('--evaluate', type=bool, default=True)

    # Dataset
    p.add_argument('--dataset', type=str, default='electricity',
                   help='One of: PeMS07 | electricity | solar_AL | MAPSS')
    p.add_argument('--window_size', type=int, default=12,
                   help='Lookback window (PeMS07:12; electricity/solar_AL:24)')
    p.add_argument('--granularity', type=int, default=5,
                   help='Minutes per step (PeMS07:5; electricity:60; solar_AL:10)')
    p.add_argument('--horizon', type=int, default=3)

    # Split ratios (for non‑MAPSS datasets)
    p.add_argument('--train_length', type=float, default=7)
    p.add_argument('--valid_length', type=float, default=2)
    p.add_argument('--test_length', type=float, default=1)

    # Optimization
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--multi_layer', type=int, default=5)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--validate_freq', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--norm_method', type=str, default='z_score')
    p.add_argument('--optimizer', type=str, default='RMSProp')
    p.add_argument('--early_stop', type=bool, default=False)
    p.add_argument('--exponential_decay_step', type=int, default=5)
    p.add_argument('--decay_rate', type=float, default=0.7)
    p.add_argument('--dropout_rate', type=float, default=0.2)
    p.add_argument('--leakyrelu_rate', type=float, default=0.2)

    # Missingness simulation & topology
    p.add_argument('--missing_rate', type=float, default=0.1,
                   help='Node dropout ratio per batch: 0 | 0.1 | 0.2 | 0.5')
    p.add_argument('--node_count', type=int, default=21)

    # Reproducibility
    p.add_argument('--seed', type=int, default=20)

    # Output root
    p.add_argument('--out_root', type=str, default='output_testloss')

    return p.parse_args()


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_data(args: argparse.Namespace):
    """Load dataset and return (train, valid, test) objects.

    - For PeMS07/electricity/solar_AL: returns np.ndarray splits (T,N).
    - For MAPSS: returns DataLoaders (already split).
    """
    data = None
    if args.dataset == 'PeMS07':
        print('Load PeMS07 data..')
        args.window_size = 12
        args.granularity = 5
        data = load_PeMS07(args)
        tr, va, te = split_data(args, data)
        print('TRAIN->', tr.shape, 'VALID->', va.shape, 'TEST->', te.shape)
        return tr, va, te

    if args.dataset == 'electricity':
        print('Load Electricity data..')
        args.window_size = 24
        args.granularity = 60
        data = load_elec(args)
        tr, va, te = split_data(args, data)
        print('TRAIN->', tr.shape, 'VALID->', va.shape, 'TEST->', te.shape)
        return tr, va, te

    if args.dataset == 'solar_AL':
        print('Load Solar_AL data..')
        args.window_size = 24
        args.granularity = 10
        data = load_solar(args)
        tr, va, te = split_data(args, data)
        print('TRAIN->', tr.shape, 'VALID->', va.shape, 'TEST->', te.shape)
        return tr, va, te

    if args.dataset == 'MAPSS':
        print('Load MAPSS data..')
        # Returns (train_loader, valid_loader, test_loader)
        return load_MAPSS(args)

    raise ValueError('Please specify available dataset: PeMS07 | electricity | solar_AL | MAPSS')


def main():
    args = parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA not available; falling back to cpu')
        args.device = 'cpu'

    # Reproducibility
    set_seed(args.seed)

    # Output directories
    result_train_file = os.path.join(args.out_root, args.dataset, str(args.missing_rate), 'train')
    result_test_file  = os.path.join(args.out_root, args.dataset, str(args.missing_rate), 'test')
    os.makedirs(result_train_file, exist_ok=True)
    os.makedirs(result_test_file,  exist_ok=True)

    # Load data once
    train_data, valid_data, test_data = prepare_data(args)

    # Training
    if args.train:
        try:
            t0 = datetime.now().timestamp()
            _, _ = train(train_data, valid_data, args, result_train_file)
            t1 = datetime.now().timestamp()
            print(f'Training took {(t1 - t0) / 60:.2f} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    # Evaluation at multiple horizons
    if args.evaluate:
        print(f"Missing rate: {args.missing_rate}, MA GRUD StemGNN")
        t0 = datetime.now().timestamp()
        for H in (3, 6, 9):
            print(f'============ horizon={H} ===========')
            args.horizon = H
            _ = test(test_data, args, result_train_file, result_test_file)
        t1 = datetime.now().timestamp()
        print(f'Evaluation took {(t1 - t0) / 60:.2f} minutes')
        args.horizon = 3  # reset default

    print('done')


if __name__ == '__main__':
    main()
