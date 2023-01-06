from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import isdir, split
from torch import save, nn


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


def main() -> None:
    """
    Create an untrained neural network

    :return: None
    """

    parser = ArgumentParser(prog="create_neural_network",
                            description="Main neural network training script",
                            add_help=True,
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument("PATH_and_filename_of_AI",
                        help=" PATH with the file name of the serialized AI being generated.\n"
                             " E.g. C:\\folder\\AI.pt\n"
                             " Note that the .pt file extension isn't strictly required but\n"
                             " it's safer to include it anyway.")

    args = parser.parse_args()

    directory, filename = split(args.PATH_and_filename_of_AI)
    if not isdir(directory):
        raise FileNotFoundError(f" The directory {directory} does not exist.\n"
                                f" Use the --help option to see the list of command line arguments.")

    # Define layer details and initialize the neural network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    neural_network = neural_model(input_size, hidden_sizes, output_size)

    # Serialize and save the untrained AI
    save(neural_network, args.PATH_and_filename_of_AI)


if __name__ == "__main__":
    main()
