# Import basic pkg
from torch.utils.data import DataLoader

# Import custom pkg
from dataset.dataset import get_datasets

# Get dataloaders
def get_dataloaders(args):

    # Datasets
    data_set_select = get_datasets(args)
    
    data_loader_select = {
        'train_aug': DataLoader(dataset = data_set_select['train_aug'],
                            shuffle=args.shuffle,
                            batch_size = args.batch_size,
                            num_workers = args.num_workers),
        'validation': DataLoader(dataset = data_set_select['validation'],
                            shuffle=False,
                            batch_size = args.batch_size,
                            num_workers = args.num_workers)
    }
    return data_loader_select