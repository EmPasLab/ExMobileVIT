# Import basic pkg
from torchvision import transforms

# Transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transform for ImageNet dataset
# Train dataset use resizedcrop/horizontalflip
transform_imagenet_train = transforms.Compose([
            transforms.RandomResizedCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize])

# validation dataset use resize
transform_imagenet_val = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize])