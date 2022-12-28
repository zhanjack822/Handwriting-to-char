from argparse import ArgumentParser, RawTextHelpFormatter
from logging import DEBUG, getLogger, info, StreamHandler
from math import floor
from os.path import isdir, isfile
from sys import stdout
from time import process_time
from torch import exp, load, nn, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def evaluate_neural_network(neural_network_model: nn, control_data: datasets) -> None:
    """
    This function logs the accuracy of the neural network's predictions against some set of control data

    :param control_data:
    :param neural_network_model:
    :return: None
    """
    start = process_time()
    correct_count, all_count = 0, 0
    for images, labels in control_data:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            # Turn off gradients to speed up this part
            with no_grad():
                log_ps = neural_network_model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = exp(log_ps)
            probability = list(ps.numpy()[0])
            prediction_label = probability.index(max(probability))
            true_label = labels.numpy()[i]
            if true_label == prediction_label:
                correct_count += 1
            all_count += 1

    info(f"Number Of Images Tested = {all_count}")
    info(f"Model Accuracy = {correct_count / all_count}")
    end = process_time()
    duration_min = floor((end - start) / 60)
    duration_sec = end - start - duration_min * 60
    info(f"Test duration: {duration_min} minutes and {duration_sec} seconds")


def main() -> None:
    """
    Load a neural network from a .pt file and test the accuracy of its predictions.

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
                        help=" PATH with the file name of the serialized AI to be loaded.\n"
                             " E.g. C:\\folder\\AI.pt\n"
                             " Note that the .pt file extension isn't strictly required but\n"
                             " it's safer to include it anyway.")
    parser.add_argument("Data_loading_subprocesses",
                        help=" Number of subprocesses used when loading and training from the data")
    parser.add_argument("pin_memory",
                        help=" Set to True if you are using a GPU, otherwise set to False")
    args = parser.parse_args()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    if not isdir(args.MNIST_directory):
        raise FileNotFoundError(f" The directory {args.MNIST_directory} does not exist.\n"
                                f" Use the --help option to see the list of command line arguments.")

    if not isfile(args.PATH_and_filename_of_AI):
        raise FileNotFoundError(f" The file {args.PATH_to_serialized_AI} does not exist.\n" 
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

    # Load neural network
    neural_network = load(args.PATH_and_filename_of_AI)

    # Define datasets for testing efficacy of trained AI
    val_set = datasets.MNIST(args.MNIST_directory, download=True, train=False, transform=transform)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=subprocesses, pin_memory=pin_memory)
    evaluate_neural_network(neural_network, val_loader)


if __name__ == "__main__":
    main()
