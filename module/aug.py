from torchvision import transforms


def horizontal_vertical_flip(prob):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(prob),
        transforms.RandomVerticalFlip(prob)
    ])