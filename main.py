# Import basic pkg
import sys, os
from datetime import datetime
from torchinfo import summary

# Add addtional pkg path
prjdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prjdir)

# Import custom pkg
from utils.parser import get_args
from utils.logger import get_logger
from utils.trainer import get_trainer
from dataset.dataloader import get_dataloaders
from models.model_utils import Classifier

# Main
def main(args):

    # Logger
    CSVLogger, TB_logger = get_logger(args)

    # Model
    model = Classifier(args, TB_logger)
    
    if args.mode == 'summary':
        # Print summary
        summary(model, (3, 256, 256), device = 'cpu')
        raise Exception("Model Printed")
    
    elif args.mode == 'print':
        # Print model structure
        print(model)
        raise Exception("Model Printed")

    # Trainer
    trainer = get_trainer(args, CSVLogger, TB_logger)

    # Dataloaders
    data_loader = get_dataloaders(args)

    if os.path.exists(os.path.join(args.log_dir, args.log_name, ('version_' + str(args.version)), args.checkpoint)):
        check_point = os.path.join(args.log_dir, args.log_name, ('version_' + str(args.version)), args.checkpoint)
    else:
        check_point = None
        
    if args.mode == 'train':
        # Start training
        trainer.fit(model, data_loader['train_aug'], data_loader['validation'], ckpt_path = check_point)

    else:
        raise Exception(f'Mode "{args.mode}" not supported.')

if __name__ == '__main__':

    # Loading parser
    args = get_args(sys.stdin)

    main(args)
