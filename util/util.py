from __future__ import print_function
import torch
import torchvision
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
# Li Jiaxin, because of the tanh() of the generator, here this function assumes that the output is in [-1,1]
# but actually for depth estimation, this function is used only for input rgb, hence is the inverse of ([0-1]-0.5)/0.5
# in this function, CxHxW -> HxWxC
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1,2,0)) * 0.5 + 0.5) * 255.0
    return image_numpy.astype(imtype)

# Li Jiaxin, for test images
def tensor2grid_im(image_tensor):
    grid = torchvision.utils.make_grid(image_tensor, nrow=5, normalize=False)
    grid = (grid.cpu().float().numpy().transpose((1,2,0)) * 0.5 + 0.5) * 255.0
    return grid.astype(np.uint8)

# Li Jiaxin, define a function to convert log depth to uint8 img
# the log depth is single channel tensor ranges around [-10, 10]
def log_depth2im(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    minimum = np.amin(image_numpy)
    maximum = np.amax(image_numpy)
    image_numpy = (np.transpose(image_numpy, (1,2,0)) - minimum) / (maximum-minimum) * 255
    return image_numpy.astype(np.uint8).repeat(3,2)

def log_depth2grid_im(image_tensor):
    # the clamp is according to the data processing
    image_tensor = image_tensor.clamp(-0.84, 2.11)
    grid = torchvision.utils.make_grid(image_tensor, nrow=5, normalize=True)
    grid = grid.cpu().float().numpy().transpose((1, 2, 0)) * 255
    return grid.astype(np.uint8)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
