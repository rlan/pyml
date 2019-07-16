from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset


def load_mnist(
    batch_size,
    test_batch_size,
    data_folder='./data',
    num_workers=2
):
    """Return DataLoaders for the MNIST dataset.
    60000 train / 10000 test.
    Reference: http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    batch_size : int
        Mini-batch size for training set.
    test_batch_size : int
        Mini-batch size for test set.
    data_folder : str
        Default: ./data. Path for saving the dataset files.
    num_workers : int
        Default: 2 (if CPU) and 4 (if GPU exists). 
        Check torch.utils.data.DataLoader for documentations.

    Returns
    -------
    train_loader, test_loader : tuple
        Torch DataLoader that contains respective datasets.
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_folder, train=True,
                       download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_folder, train=False, transform=transform),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs)
    return train_loader, test_loader


def load_mnist_with_validation(
    batch_size,
    test_batch_size,
    validation_size,
    data_folder='./data',
    num_workers=2
):
    """Return DataLoaders for the MNIST dataset.
    Forms a validation dataset from the last 'validation_size' images of the 
    training set. Rest of the images are return as the training set. The
    original test set is unmodified.

    Reference: http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    batch_size : int
        Mini-batch size for training set.
    test_batch_size : int
        Mini-batch size for test set.
    validation_size : int
        Number of images for validation set.
    data_folder : str
        Default: ./data. Path for saving the dataset files.
    num_workers : int
        Default: 2 (if CPU) and 4 (if GPU exists). 
        Check torch.utils.data.DataLoader for documentations.

    Returns
    -------
    train_loader, validation_loader, test_loader : tuple
        Torch DataLoader that contains respective datasets.
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(
        data_folder, train=True, download=True, transform=transform)
    assert validation_size < len(train_data)
    test_data = datasets.MNIST(data_folder, train=False, transform=transform)
    validation_split = len(train_data) - validation_size
    train_dataset = Subset(train_data, range(0, validation_split))
    validation_dataset = Subset(train_data, range(
        validation_split, len(train_data)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs)
    return train_loader, validation_loader, test_loader


def get_cifar10_transforms(use_data_augmentation):
    """Return data transforms.

    Optionally, choose to use data augmentation. The method of data
    augmentation is used in the following paper:

    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 
    In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778). 
    IEEE. https://doi.org/10.1109/CVPR.2016.90

    Parameters
    ----------
    use_data_augmentation : bool
        Controls whether to use data augmentation.

    Returns
    -------
    transform : torchvision.transforms
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ])
    if use_data_augmentation:
        return transform_train, transform_test
    else:
        return transform_test, transform_test


def load_cifar10(
    batch_size,
    test_batch_size,
    use_data_augmentation=False,
    data_folder='./data',
    num_workers=2
):
    """Return DataLoaders for the CIFAR-10 dataset.
    50000 train / 10000 test.

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html

    Parameters
    ----------
    batch_size : int
        Mini-batch size for training set.
    test_batch_size : int
        Mini-batch size for test set.
    use_data_augmentation : bool
        Default: False. Use data augmentation. See source code for method.
    data_folder : str
        Default: ./data. Path for saving the dataset files.
    num_workers : int
        Default: 2 (if CPU) and 4 (if GPU exists). 
        Check torch.utils.data.DataLoader for documentations.

    Returns
    -------
    train_loader, test_loader : tuple
        Torch DataLoader that contains respective datasets.
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(
        use_data_augmentation)
    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_cifar10_with_validation(
    batch_size,
    test_batch_size,
    validation_size,
    use_data_augmentation=False,
    data_folder='./data',
    num_workers=2,
):
    """Return DataLoaders for the CIFAR-10 dataset.
    50000-'validation_size' train / 'validation_size' validation / 10000 test.

    Forms a validation dataset from the last 'validation_size' images of the 
    training set. Rest of the images are return as the training set. The
    original test set is unmodified.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for training set.
    test_batch_size : int
        Mini-batch size for test set.
    validation_size : int
        Number of images for validation set.
    use_data_augmentation : bool
        Default: False. Use data augmentation. See source code for method.
    data_folder : str
        Default: ./data. Path for saving the dataset files.
    num_workers : int
        Default: 2 (if CPU) and 4 (if GPU exists). 
        Check torch.utils.data.DataLoader for documentations.

    Returns
    -------
    train_loader, validation_loader, test_loader : tuple
        Torch DataLoader that contains respective datasets.
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(
        use_data_augmentation)

    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    assert (validation_size < len(train_data))
    sets = [len(train_data)-validation_size, validation_size]
    print("Dataset splits {}".format(sets))

    validation_split = len(train_data) - validation_size
    train_dataset = Subset(train_data, range(0, validation_split))
    validation_dataset = Subset(train_data, range(
        validation_split, len(train_data)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader


def load_cifar10_with_preselected_validation(
    batch_size,
    test_batch_size,
    index_json_file,
    use_data_augmentation=False,
    data_folder='./data',
    num_workers=2,
):
    """Return DataLoaders for the CIFAR-10 dataset.
    50000-'validation_size' train / 'validation_size' validation / 10000 test.

    Forms a validation dataset from the last 'validation_size' images of the 
    training set. Rest of the images are return as the training set. The
    original test set is unmodified.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for training set.
    test_batch_size : int
        Mini-batch size for test set.
    index_json_file : str
        A json file that contains the indices for training and validation set.
        Use generate_train_val_indices() to generate one.
    use_data_augmentation : bool
        Default: False. Use data augmentation. See source code for method.
    data_folder : str
        Default: ./data. Path for saving the dataset files.
    num_workers : int
        Default: 2 (if CPU) and 4 (if GPU exists). 
        Check torch.utils.data.DataLoader for documentations.

    Returns
    -------
    train_loader, validation_loader, test_loader : tuple
        Torch DataLoader that contains respective datasets.
    """
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {'num_workers': num_workers}
    transform_train, transform_test = get_cifar10_transforms(
        use_data_augmentation)

    train_indices, val_indices = load_train_val_indices(index_json_file)
    train_data = datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train)
    assert (len(train_data) == len(train_indices) + len(val_indices))
    train_dataset = Subset(train_data, train_indices)
    validation_data = datasets.CIFAR10(
        root=data_folder, train=True, download=False, transform=transform_test)
    validation_dataset = Subset(validation_data, val_indices)
    test_data = datasets.CIFAR10(
        root=data_folder, train=False, transform=transform_test)
    print("Dataset splits {}/{}/{}".format(len(train_indices),
                                           len(val_indices), len(test_data)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader


def generate_train_val_indices(input_size, val_size, file_name):
    """Generate and save to file random indices for training and validation set.

    Parameters
    ----------
    input_size : int
        Dataset size.
    val_size : int
        Validation size.
    file_name : str
        Output json file name.

    Returns
    -------
    train_indices : list(int)
    val_indices : list(int)
    """
    assert (val_size < input_size)
    indices = [x for x in range(0, input_size)]
    random.shuffle(indices)
    train_indices = indices[val_size:]
    validation_indices = indices[:val_size]
    print("Indices generated: train size {} val size {}".format(
        len(train_indices), len(validation_indices)))

    # Write indices to a json file
    with open(file_name, 'w') as fp:
        data = {'train': train_indices, 'validation': validation_indices}
        json.dump(data, fp)
        print("Saved to {}...".format(file_name))

    return train_indices, validation_indices


def load_train_val_indices(file_name):
    """Read from file indices for training and validation set.

    Parameters
    ----------
    file_name : str
        Output json file name.

    Returns
    -------
    train_indices : list(int)
    val_indices : list(int)
    """

    with open(file_name, 'r') as fp:
        print("Loading from {}...".format(file_name))
        data = json.load(fp)
        return data['train'], data['validation']
