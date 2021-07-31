import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Configuration and Parameters
    # for anomaly detection
    parser.add_argument('--w1', type=int, default=1800,
        help='Windows for testing/detecting before the injection time.')
    # parser.add_argument('--w2', type=int, default=3600,
    #     help='The interval for sampling normal points before change start time')
    # parser.add_argument('--j-window', type=int, default=6,
    #     help='')
    # parser.add_argument('--ananomly-th', type=int, default=5,
    #     help='')
    parser.add_argument('--ad-method', type=str, choices=['abs', 'mad', 'spot'], default='abs',
        help='Method of anomaly detection')
    # for page rank
    parser.add_argument('--pr-alpha', type=float, default=0.85,
        help='non-teleport proba for pagerank')
    parser.add_argument('--pr-eps', type=float, default=1e-3,
        help='convergence constant for pagerank')

    parser.add_argument('--seed', type=int, default=0,
        help='Random seed')

    # Log and Output
    parser.add_argument('--train_dir', type=str, default='./train',
        help='Training directory for saving model. Default: ./train')
    parser.add_argument('--continue_train', type=int, default=0,
        help='Do or not continue to train')

    # Data
    parser.add_argument('--data-folder', type=str, default='../../aiops-2021/data/system_kpis',
        help='Training system kpi data folder')
    parser.add_argument('--injection-times', type=str, default='../../aiops-2021/ground_truth/injection_times.csv',
        help='Injection time input')
    parser.add_argument('--service-kpi', type=str, default='../../aiops-2021/data/service_kpi.csv',
        help='Training service kpi data folder')


    # Checkpointing
    # parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    # parser.add_argument("--save-dir", type=str, default="./model/", help="directory in which training state and model should be saved")
    # parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    # parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # # Evaluation
    # parser.add_argument("--restore", action="store_true", default=False)
    # parser.add_argument("--display", action="store_true", default=False)
    # parser.add_argument("--benchmark", action="store_true", default=False)
    # parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    # parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    # parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
