from torch import exp, no_grad, save
from torch.utils.data import DataLoader
from torch.jit import trace
from argparse import ArgumentParser, RawTextHelpFormatter
from torch import nn, optim
from torchvision import datasets, transforms
from os.path import isdir, split
from time import perf_counter
from logging import info, DEBUG, getLogger, StreamHandler
from sys import stdout


def neural_model(input_layer: int, hidden_layer_sizes: list, output_layer: int) -> nn.Sequential:
    """
    Initialize the neural network with four layers of neurons and the size of each layer is defined by
    the function parameters

    :param input_layer: The size of the first layer of neurons
    :param hidden_layer_sizes: The size of the two "hidden layers" of neurons between the input and output layers
    :param output_layer: The size of the output layer of neurons
    :return: model - a container for several neural network models which will cumulatively form the machine learning
    algorithm
    :raise ValueError: The number of elements in hidden_sizes isn't two
    """

    if len(hidden_layer_sizes) != 2:
        raise ValueError(f"Expected hidden_sizes to be a list with two elements, received {len(hidden_layer_sizes)}")

    model = nn.Sequential(nn.Linear(input_layer, hidden_layer_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_layer_sizes[1], output_layer),
                          nn.LogSoftmax(dim=1))

    return model


def nn_training(epochs: int, neural_network_model: nn, training_data: datasets, criterion: nn.NLLLoss) -> None:
    """
    This function trains the neural network over several epochs using some training data and a criterion for
    evaluating the neural network's accuracy in its predictions.

    :param epochs: The number of iterations used to train the neural network
    :param training_data: The dataset used to train the neural network
    :param neural_network_model: The neural network being trained
    :param criterion: The criterion that the training algorithm uses to measure the efficacy of the neural network's
    accuracy in each iterative loop
    :return: None
    """
    optimizer = optim.SGD(neural_network_model.parameters(), lr=0.003, momentum=0.9)
    start = perf_counter()
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
    end = perf_counter()
    duration = (end - start) / 60  # in minutes
    info(f"Training Time (in minutes) = {duration}")


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
    Create a trained neural network AI, serialize it, and save it to the root directory of the project.

    TODO: Restructure script using TorchScript and PyTorch's JIT to speed up the script

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
                        help=" PATH with the file name of the serialized AI being generated.\n"
                             " E.g. C:\\folder\\AI.pt\n"
                             " Note that the .pt file extension isn't strictly required but\n"
                             " it's safer to include it anyway.")
    args = parser.parse_args()

    if not isdir(args.MNIST_directory):
        raise FileNotFoundError(f"""The directory {args.MNIST_directory} does not exist. \n Use the --help option
                                    to see the list of command line arguments.""")

    directory, filename = split(args.PATH_and_filename_of_AI)
    if not isdir(directory):
        raise FileNotFoundError(f"""The directory {directory} does not exist. \n Use the --help option
                                    to see the list of command line arguments.""")

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
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Layer details for the neural network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    neural_network = neural_model(input_size, hidden_sizes, output_size)

    # The cross entropy loss of a neural network is its error or accuracy, and ranges from 0 to 1, with 0 being a
    # perfectly accurate model while a 1 is always wrong. In order to train the AI, we need to define the loss of the
    # model in each iterative loop it executes to train itself from the data, so it knows how to improve.
    criterion = nn.NLLLoss()

    # train the neural network
    epochs = 15
    nn_training(epochs, neural_network, train_loader, criterion)

    # Serialize trained AI after training
    save(neural_network, args.PATH_and_filename_of_AI)


if __name__ == "__main__":
    main()
