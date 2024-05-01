# Import custom pkg
from models.MobileViT import MobileViT
from models.ExMobileViT_576 import ExMobileViT as ExMobileViT_576
from models.ExMobileViT_640 import ExMobileViT as ExMobileViT_640
from models.ExMobileViT_928 import ExMobileViT as ExMobileViT_928

# If you want to add custom model, put in here.

def get_model(args):
    model_args = making(args)

    if model_args['model_str'] in ['mvit']:
        return MobileViT(image_size = model_args['input_size'],
                         dims = model_args['dims'],
                         channels = model_args['channels'], 
                         num_classes = model_args['num_classes'], )
    
    elif model_args['model_str'] in ['ExMobileViT-928']:
        return ExMobileViT_928(image_size = model_args['input_size'],
                    dims = model_args['dims'],
                    channels = model_args['channels'], 
                    num_classes = model_args['num_classes'], )
    
    elif model_args['model_str'] in ['ExMobileViT-640']:
        return ExMobileViT_640(image_size = model_args['input_size'],
                    dims = model_args['dims'],
                    channels = model_args['channels'], 
                    num_classes = model_args['num_classes'], )
    
    elif model_args['model_str'] in ['ExMobileViT-576']:
        return ExMobileViT_576(image_size = model_args['input_size'],
                    dims = model_args['dims'],
                    channels = model_args['channels'], 
                    num_classes = model_args['num_classes'], )
    
    # Put in here
    #elif model_args['model_str'] in ['Put in here']:
    #    return NewModel(image_size = model_args['input_size'],
    #                dims = model_args['dims'],
    #                channels = model_args['channels'], 
    #                num_classes = model_args['num_classes'], )

def making(args):
    model_str = args.architecture

    if model_str in ['mvit', 'ExMobileViT-928', 'ExMobileViT-640', 'ExMobileViT-576']:
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        input_size = (256, 256)
        num_classes = args.n_classes

    #elif model_str in ['NewModel']:
    #    dims = []
    #    channels = []
    #    input_size = (256, 256)
    #    num_classes = args.n_classes

    else:
        raise Exception(f'Model "{model_str}" not supported.')

    data_loader_select = {
        'model_str': model_str,
        'dims': dims,
        'channels': channels,
        'input_size': input_size,
        'num_classes': num_classes,
    }

    return data_loader_select