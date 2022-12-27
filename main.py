import torch.utils.data
from argparse import ArgumentParser
from torch import nn, optim
from torchvision import datasets, transforms
from os.path import isdir

if __name__ == "__main__":
    parser = ArgumentParser(prog="main",
                            description="Main neural network training script",
                            add_help=True)
    parser.add_argument("MNIST_directory",
                        help="""Path or relative path to root directory where MNIST dataset is located.\n 
                        If the dataset hasn't been downloaded yet, this is the directory where the data will be
                        installed in.""")
    args = parser.parse_args()

    if not isdir(args.MNIST_directory):
        raise FileNotFoundError(f"""The directory {args.MNIST_directory} does not exist. \n Use the --help option
                                to see the list of command line arguments.""")

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    # If the data has not been downloaded yet, it will be downloaded to the directory specified in the command line
    # argument and referenced by several variables we declare. If the data is already downloaded, the installation
    # is skipped.
    train_set = datasets.MNIST(args.MNIST_directory, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(args.MNIST_directory, download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
