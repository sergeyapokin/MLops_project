import argparse
from source.prepare_data import load_dataset
from train import train
from infer import infer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', type=str,
                        help="train/infer/both(train and then infer)")
    parser.add_argument('--lookback', default=3, type=int)
    parser.add_argument('--path_to_save_model', default='lightning_logs/checkpoints/model.ckpt', type=str)
    parser.add_argument('--trained_model', default='lightning_logs/checkpoints/model.ckpt', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    _, X_train, y_train, X_test, y_test = load_dataset(args.lookback)
    if args.stage == 'train':
        trained_model = train(X_train, y_train, args.path_to_save_model)
    elif args.stage == 'infer':
        y_pred = infer(X_test, y_test, args.trained_model)
    elif args.stage == 'both':
        trained_model = train(X_train, y_train, args.path_to_save_model)
        y_pred = infer(X_test, y_test, trained_model=trained_model)
    else:
        raise NotImplementedError
