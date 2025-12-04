from matplotlib import transforms
import torch
import os
import torch
import numpy as np
import torchvision.utils as vutils
import copy
import random
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2 

from tqdm import tqdm

from deepinv.tests.dummy_datasets import datasets


def set_seed(seed: int):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_metrics(save_path):

    images_path = save_path + "/images"
    model_path = save_path + "/model"
    metrics_path = save_path + "/metrics"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return images_path, model_path, metrics_path


def save_npy_metric(file, metric_name):

    with open(f"{metric_name}.npy", "wb") as f:
        np.save(f, file)


def save_coded_apertures(
    system_layer,
    row,
    pad,
    path,
    name,
):

    aperture_codes = copy.deepcopy(system_layer.get_mask().detach()).permute(1, 0, 2, 3)

    grid = vutils.make_grid(aperture_codes, nrow=row, padding=pad, normalize=True)
    vutils.save_image(grid, f"{path}/{name}.png")

    return grid


def save_reconstructed_images(imgs, recons, num_img, pad, path, name, PSNR, SSIM):

    grid = vutils.make_grid(
        torch.cat((imgs[:num_img], recons[:num_img])), nrow=num_img, padding=pad, normalize=True
    )
    vutils.save_image(grid, f"{path}/{name}.png")

    psnr_imgs = [
        np.round(PSNR(recons[i].unsqueeze(0), imgs[i].unsqueeze(0)).item(), 2)
        for i in range(num_img)
    ]
    ssim_imgs = [
        np.round(SSIM(recons[i].unsqueeze(0), imgs[i].unsqueeze(0)).item(), 3)
        for i in range(num_img)
    ]

    return grid, psnr_imgs, ssim_imgs


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")



class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png') or img.endswith('.jpg')]
        # Preload all images into RAM as tensors
        self.images = []
        # Use multithreading to speed up image loading

        def load_image(img_path):
            with Image.open(img_path) as image:
                image = image.convert('L')  # or 'RGB'
                if self.transform:
                    image = self.transform(image)
            return image

        with ThreadPoolExecutor() as executor:
            self.images = list(tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths), desc="Loading images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    

class JPGImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [img for img in os.listdir(image_dir) if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image =  Image.fromarray(np.array(cv2.imread(img_path))).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Assuming labels are encoded in the file name or directory structure, modify as needed
        
        return image


    
def get_dataloaders(args):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((args.n,args.n), antialias=True),transforms.Grayscale()])
    if args.dataset == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'fashionmnist':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'cifar10':        
        root = './data'
        already = os.path.isdir(os.path.join(root, 'cifar-10-batches-py'))
        trainset = datasets.CIFAR10(root=root, train=True, download=not already, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=root, train=False, download=not already, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        
    
    elif args.dataset == 'CelebA':
        dataset = JPGImageDataset(r'./data/CelebA', transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = random_split(dataset, [train_size, test_size])
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
   
    # elif args.dataset == 'cifar10':
    #     root = './data'
    #     already = os.path.isdir(os.path.join(root, 'cifar-10-batches-py'))
    #     trainset = datasets.CIFAR10(root=root, train=True, download=not already, transform=transform)
    #     testset = datasets.CIFAR10(root=root, train=False, download=not already, transform=transform)
    #     test_size = 10000
    #     val_size = 10000
    #     train_size = 40000

    #     trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_size, shuffle=False)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=False)
    #     valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=False)
   
    return trainloader, testloader



def psnr_fun(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula el PSNR de cada muestra en el batch y su media.
    Si mse_per_sample == 0 → PSNR = +∞
    Si peak == 0     → PSNR = -∞

    Args:
        preds: Tensor de predicciones, forma (N, C, H, W).
        target: Tensor de ground truth, forma (N, C, H, W).
        use_dynamic_range: si True, usa (max-min) por muestra; 
                           si False, usa solo max (asumiendo min=0).

    Returns:
        psnr_per_sample: Tensor (N,) con el PSNR de cada muestra.
        psnr_mean: escalar con el PSNR medio sobre el batch.
    """
    N = preds.shape[0]
    flat_preds  = preds.view(N, -1)
    flat_target = target.view(N, -1)

    # 1) MSE por muestra
    mse_per_sample = torch.mean((flat_preds - flat_target) ** 2, dim=1)  # (N,)

    # 2) Pico por muestra
    max_val = flat_target.max(dim=1).values
    min_val = flat_target.min(dim=1).values
    peak = max_val - min_val

    # 3) PSNR por muestra, sin eps para permitir ±∞
    psnr_per_sample = 10 * torch.log10(peak**2 / mse_per_sample)

    # 4) Media sobre el batch (ignorando infinities en la media de PyTorch)
    psnr_mean = psnr_per_sample.mean()

    return psnr_mean
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png') or img.endswith('.jpg')]
        # Preload all images into RAM as tensors
        self.images = []
        # Use multithreading to speed up image loading

        def load_image(img_path):
            with Image.open(img_path) as image:
                image = image.convert('L')  # or 'RGB'
                if self.transform:
                    image = self.transform(image)
            return image

        with ThreadPoolExecutor() as executor:
            self.images = list(tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths), desc="Loading images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]