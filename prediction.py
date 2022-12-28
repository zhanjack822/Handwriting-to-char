from torch import exp, no_grad, load
from torch.utils.data import DataLoader
from torch.jit import trace
from argparse import ArgumentParser, RawTextHelpFormatter
from torch import nn
from torchvision import datasets, transforms
from os.path import isdir, isfile
from time import perf_counter
from logging import info, DEBUG, getLogger, StreamHandler
from sys import stdout


def evaluate_neural_network(neural_network_model: nn, control_data: datasets) -> None:
    """
    This function logs the accuracy of the neural network's predictions against some set of control data

    :param control_data:
    :param neural_network_model:
    :return: None
    """

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


def main() -> None:
    """

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
    args = parser.parse_args()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    if not isdir(args.MNIST_directory):
        raise FileNotFoundError(f"""The directory {args.MNIST_directory} does not exist. \n Use the --help option
                                    to see the list of command line arguments.""")
    if not isfile(args.PATH_and_filename_of_AI):
        raise FileNotFoundError(f"""The file {args.PATH_to_serialized_AI} does not exist. \n Use the --help option
                                    to see the list of command line arguments.""")

    # Configure logger
    log = getLogger()
    log.setLevel(DEBUG)
    log.addHandler(StreamHandler(stdout))

    # Load neural network
    neural_network = load(args.PATH_and_filename_of_AI)

    # Define datasets for testing efficacy of trained AI
    val_set = datasets.MNIST(args.MNIST_directory, download=True, train=False, transform=transform)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    evaluate_neural_network(neural_network, val_loader)


if __name__ == "__main__":
    main()
