import math
import pandas as pd
import torch
import matplotlib.pyplot as plt

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def base_linspace_int(start, base, repeat):
    return [(start * base**i) for i in range(repeat)]


def label_convert(csv_path, mode):
    data = pd.read_csv(csv_path, header=None)
    if data.shape[1] == 2:
        labels = sorted(list(set(data.iloc[1:,1])))
        n_classes = len(labels)
        classes_to_num = dict(zip(labels, range(n_classes)))
        num_to_classes = {v: k for k, v in classes_to_num.items()}
        if mode == 'num':
            return classes_to_num
        elif mode == 'class':
            return num_to_classes
    elif data.shape[1] == 1:
        return None


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
