"""This file contains all arguments required for model training."""
import argparse

# Training settings
def get_args():
    """Primary function to retrieve arguments."""
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    # general
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    # hyperparams
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)")
    parser.add_argument("--learn-rate", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")

    # network params to add
    # None

    # save model
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    parser.add_argument("--save-name", type=str, default="mnist.pt", help="Model name")

    # pretrained weights
    parser.add_argument("--use-pretrained", action="store_true", default=False, help="use pretrained weights")
    parser.add_argument("--pretrained-model-name", type=str, default="mnist.pt", help="path to pretrained weights")

    return parser.parse_args()
