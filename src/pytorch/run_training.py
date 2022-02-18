"""This module contains the primary training / testing code for the network model."""
# Standard library imports
import os
from tempfile import gettempdir

# Other library imports
import torch
import torch.optim as optim
import torch.nn.functional as torch_fn
import numpy as np
from pytorch.network import MNISTNet
from pytorch.data import get_dataloader

# Local library imports
from utils.utils_pytorch import load_model, save_model

##### Public Functions #####
def run_training(logger, args):
    """
    Main training / testing orchestration function

    Parameters
    ----------
    logger: Logger
        Logger object for logging status messages.
    args: object
        List of arguments required to run the training/testing process.
    """
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get network and optimizer
    model = MNISTNet().to(device)
    optimizer = _get_optimizer(model, args.learn_rate, args.momentum)

    # load weights, where applicable
    if args.use_pretrained:
        model = load_model(args.pretrained_model_name)

    # get data loaders
    train_loader = get_dataloader(args.batch_size, is_train=True, to_shuffle=True)
    test_loader = get_dataloader(args.batch_size, is_train=False, to_shuffle=True)

    # train and validate
    print(f"epochs {args.epochs + 1}")
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval, logger)
        # Print separating between traing and test
        print()
        test(model, device, test_loader, epoch, logger)
        print()
        if args.save_model:
            save_model(model, os.path.join(gettempdir(), args.save_name))
        logger.current_logger().report_text(
            f"The default output destination for model snapshots and artifacts is: {args.save_name}"
        )
        print("\n")


def train(model, device, train_loader, optimizer, epoch, log_interval, logger):
    """
    Default training function as taken from ClearML examples.

    Parameters
    ----------
    model: object
        Neural network model for training.
    device: str
        "cuda" or "cpu".
    train_loader: object
        Training data.
    optimizer: object
        Neural network optimizer object.
    epoch: int
        Current epoch number.
    log_interval: int
        Determines frequency of logging messages.
    logger: Logger
        Logger object for logging status messages.
    """
    print(f"epoch {epoch}")
    save_loss = []

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()
        save_loss.append(loss)

        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.current_logger().report_scalar(
                "train", "loss", iteration=(epoch * len(train_loader) + batch_idx), value=loss.item()
            )
            _print_training_step(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                _calculate_percent(batch_idx / len(train_loader)),
                loss.item(),
            )
            # Add manual scalar reporting for loss metrics
            logger.current_logger().report_scalar(
                title=f"Scalar example {epoch} - epoch", series="Loss", value=loss.item(), iteration=batch_idx
            )


def test(model, device, test_loader, epoch, logger):
    """
    Default testing function as taken from ClearML examples.

    Parameters
    ----------
    model: object
        Neural network model for testing.
    device: str
        "cuda" or "cpu".
    test_loader: object
        Testing data.
    epoch: int
        Current epoch number.
    logger: Logger
        Logger object for logging status messages.
    """
    save_test_loss = []
    save_correct = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch_fn.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            save_test_loss.append(test_loss)
            save_correct.append(correct)

    test_loss /= len(test_loader.dataset)

    logger.current_logger().report_scalar("test", "loss", iteration=epoch, value=test_loss)
    logger.current_logger().report_scalar(
        "test", "accuracy", iteration=epoch, value=(correct / len(test_loader.dataset))
    )
    _print_test_step(
        test_loss, correct, len(test_loader.dataset), _calculate_percent(correct / len(test_loader.dataset))
    )
    logger.current_logger().report_histogram(
        title="Histogram example", series="correct", iteration=1, values=save_correct, xaxis="Test", yaxis="Correct"
    )
    # Manually report test loss and correct as a confusion matrix
    matrix = np.array([save_test_loss, save_correct])
    logger.current_logger().report_confusion_matrix(
        title="Confusion matrix example", series="Test loss / correct", matrix=matrix, iteration=1
    )


def compute_loss(output, target):
    """
    Computes the loss between prediction and actual target values.

    Parameters
    ----------
    output: object
        model predictions
    target: object
        actual target labels
    """
    return torch_fn.nll_loss(output, target)


##### Private Functions #####
def _calculate_percent(value):
    return 100.0 * value


def _print_training_step(epoch, sample_num, total_samples, percent_done, loss):
    print(f"Train Epoch: {epoch} [{sample_num}/{total_samples} ({percent_done:.0f}%)] Loss: {loss:.6f}")


def _print_test_step(loss, correct_samples, total_samples, percent_correct):
    print(f"Test set: Average loss: {loss:.4f}, Accuracy: {correct_samples}/{total_samples} ({percent_correct:.0f}%)")


def _get_optimizer(model, learn_rate, momentum):
    return optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
