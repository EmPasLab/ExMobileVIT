# Import basic pkg
import os
from torchvision.datasets import ImageNet

# Import custom pkg
from dataset.transform import transform_imagenet_train, transform_imagenet_val

# Get datasets
def get_datasets(args):

    if args.data == 'imagenet' or args.data == 'ImageNet':
        data_set_select = {
            'train_aug': ImageNet(transform = transform_imagenet_train,
                                split='train',
                                root = args.data_root),
            'validation': ImageNet(transform = transform_imagenet_val,
                                split='val',
                                root = args.data_root),
        }
    else:
        raise Exception(f'MODEL "{args.model}" not supported.')

    return data_set_select