import sys
import warnings

from abundance_aware.cli.utils import GetCLIParser, GetCFGReader
from abundance_aware.src.utils.TrainingUtils import seed_everything

warnings.filterwarnings('ignore')


def main():
    print("Starting MGM...")
    parser = GetCLIParser()
    args = parser.parse_args()
    cfg = GetCFGReader(args.config)

    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        seed_everything(args.seed)
    else:
        print("No seed provided, the program will generate a random seed.")

    if args.mode == 'construct':
        from abundance_aware.cli.main_construct import  construct
        construct(cfg, args)
        sys.exit(0)
    elif args.mode == 'pretrain':
        from abundance_aware.cli.main_pretrain import pretrain
        pretrain(cfg, args)
        sys.exit(0)
    elif args.mode == 'train':
        from abundance_aware.cli.main_train import train
        train(cfg, args)
        sys.exit(0)
    elif args.mode == 'finetune':
        from abundance_aware.cli.main_finetune import finetune
        finetune(cfg, args)
        sys.exit(0)
    elif args.mode == 'predict':
        from abundance_aware.cli.main_predict import predict
        predict(cfg, args)
        sys.exit(0)
    elif args.mode == 'generate':
        from abundance_aware.cli.main_generate import generate
        generate(cfg, args)
        sys.exit(0)
    elif args.mode == 'reconstruct':
        from abundance_aware.cli.main_reconstruct import reconstruct
        reconstruct(cfg, args)
        sys.exit(0)
    else:
        raise RuntimeError('Please specify correct work mode, see `--help`.')
