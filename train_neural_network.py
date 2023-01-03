from argparse import ArgumentParser, RawTextHelpFormatter
from logging import DEBUG, getLogger, info, StreamHandler
from math import floor
from os.path import isdir, isfile
from sys import stdout
from time import perf_counter, process_time
from torch import load, nn, optim, save
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyjion import enable as pyjion_enable


def nn_training(epochs: int, neural_network_model: nn, training_data: datasets, criterion: nn.NLLLoss) -> None:
    """
    This function trains the neural network over several epochs using some training data and a criterion for
    evaluating the neural network's accuracy in its predictions. Since the function is passed a reference
    (i.e. a pointer) to the neural network which is mutable, the neural network object outside the scope of
    this function is altered as the neural_network_model object within the function scope is altered per
    iteration of the training loop, so this function doesn't need to return a new object assigned to the
    trained neural network.

    :param epochs: The number of iterations used to train the neural network
    :param training_data: The dataset used to train the neural network
    :param neural_network_model: The neural network being trained
    :param criterion: The criterion that the training algorithm uses to measure the efficacy of the neural network's
    accuracy in each iterative loop
    :return: None
    """

    optimizer = optim.SGD(neural_network_model.parameters(), lr=0.003, momentum=0.9)
    start_elapsed = perf_counter()
    start_process = process_time()
    # Use the GPU for storing tensors if possible
    if is_available():
        info("GPU used")
        for e in range(epochs):
            running_loss = 0
            for images, labels in training_data:
                # move images and labels to GPU
                images, labels = images.cuda(), labels.cuda()

                # Flatten MNIST images into a one dimensional array
                images = images.view(images.shape[0], -1)

                # Training pass
                optimizer.zero_grad()

                output = neural_network_model(images)
                loss = criterion(output, labels)

                # This is where the model learns by back-propagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            else:
                info(f"Epoch {e} - Training loss: {running_loss / len(training_data)}")

    else:
        info("CPU used")
        for e in range(epochs):
            running_loss = 0
            for images, labels in training_data:
                # Flatten MNIST images into a one dimensional array
                images = images.view(images.shape[0], -1)

                # Training pass
                optimizer.zero_grad()

                output = neural_network_model(images)
                loss = criterion(output, labels)

                # This is where the model learns by back-propagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            else:
                info(f"Epoch {e} - Training loss: {running_loss / len(training_data)}")
    end_elapsed = perf_counter()
    end_process = process_time()
    elapsed_time = end_elapsed - start_elapsed  # in seconds
    processor_time = end_process - start_process  # in seconds
    info(f"Training elapsed time: {floor(elapsed_time / 60)} minutes and {elapsed_time % 60} seconds")
    info(f"Training processor time: {floor(processor_time / 60)} minutes and {processor_time % 60} seconds")


def main() -> None:
    """
    Load a neural network from a .pt file, train it, and overwrite the .pt file with the trained neural network

    :return: None
    """

    parser = ArgumentParser(prog="main",
                            description="Main neural network training script",
                            add_help=True,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("MNIST_directory",
                        help=" Path or relative path to root directory where MNIST dataset\n"
                             " is located. If the dataset hasn't been downloaded yet,\n"
                             " this is the directory where the data will be installed in.")
    parser.add_argument("PATH_and_filename_of_AI",
                        help=" PATH with the file name of the serialized AI being trained.\n"
                             " E.g. C:\\folder\\AI.pt\n"
                             " Note that the .pt file extension isn't strictly required but\n"
                             " it's safer to include it anyway.")
    parser.add_argument("Data_loading_subprocesses",
                        help=" Number of subprocesses used when loading and training from the data")
    parser.add_argument("pin_memory",
                        help=" Set to True if you are using a GPU, otherwise set to False")
    args = parser.parse_args()

    if not isdir(args.MNIST_directory):
        raise FileNotFoundError(f" The directory {args.MNIST_directory} does not exist.\n "
                                f" Use the --help option to see the list of command line arguments.")

    if not isfile(args.PATH_and_filename_of_AI):
        raise FileNotFoundError(f" The file {args.PATH_and_filename_of_AI} does not exist.\n"
                                f" Use the --help option to see the list of command line arguments.")

    subprocesses = int(args.Data_loading_subprocesses)
    if subprocesses < 0:
        raise ValueError(f" The number of subprocesses must be greater than or equal to zero,"
                         f" but is currently set to {subprocesses}")

    pin_memory = bool(args.pin_memory)

    # Configure logger
    log = getLogger()
    log.setLevel(DEBUG)
    log.addHandler(StreamHandler(stdout))

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # If the data has not been downloaded yet, it will be downloaded to the directory specified in the command line
    # argument and referenced by several variables we declare. If the data is already downloaded, the installation
    # is skipped.
    train_set = datasets.MNIST(args.MNIST_directory, download=True, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=subprocesses, pin_memory=pin_memory)

    # Load neural network
    neural_network = load(args.PATH_and_filename_of_AI)

    # The cross entropy loss of a neural network is its error or accuracy, and ranges from 0 to 1, with 0 being a
    # perfectly accurate model while a 1 is always wrong. In order to train the AI, we need to define the loss of the
    # model in each iterative loop it executes to train itself from the data, so it knows how to improve.
    criterion = nn.NLLLoss()

    # train the neural network
    epochs = 15
    nn_training(epochs, neural_network, train_loader, criterion)

    # Overwrite serialized AI with trained AI
    save(neural_network, args.PATH_and_filename_of_AI)


if __name__ == "__main__":
    pyjion_enable()
    main()
