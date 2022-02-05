# TODO: to find a more correct way to fix import issue.
import os
import sys
_current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_current_path, '..'))


import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from network import MNISTNet
from data import get_dataloader
from utils.utils_pytorch import load_model, save_model, write_to_tensorboard
from tempfile import gettempdir


def train(model, device, train_loader, optimizer, epoch, log_interval, logger):
    print('epoch {}'.format(epoch))
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
                "train", "loss", iteration=(epoch * len(train_loader) + batch_idx), value=loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
             # Add manual scalar reporting for loss metrics
            logger.current_logger().report_scalar(title='Scalar example {} - epoch'.format(epoch), 
                series='Loss', value=loss.item(), iteration=batch_idx)


def test(model, device, test_loader, epoch, logger):
    save_test_loss = []
    save_correct = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            save_test_loss.append(test_loss)
            save_correct.append(correct)

    test_loss /= len(test_loader.dataset)

    logger.current_logger().report_scalar(
        "test", "loss", iteration=epoch, value=test_loss)
    logger.current_logger().report_scalar(
        "test", "accuracy", iteration=epoch, value=(correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.current_logger().report_histogram(title='Histogram example', series='correct',
        iteration=1, values=save_correct, xaxis='Test', yaxis='Correct')
     # Manually report test loss and correct as a confusion matrix
    matrix = np.array([save_test_loss, save_correct])
    logger.current_logger().report_confusion_matrix(title='Confusion matrix example', 
        series='Test loss / correct', matrix=matrix, iteration=1)


def get_optimizer(model, lr, momentum):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def compute_loss(output, target):
    # in complex case, some codes needed to extract/process data before computing the loss, thus keep it in a method
    # this method will also define the loss function
    return F.nll_loss(output, target)


def run_training(logger, args):
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get network and optimizer
    model = MNISTNet().to(device)
    optimizer = get_optimizer(model, args.lr, args.momentum)

    # load weights, where applicable
    if args.use_pretrained:
        model = load_model(args.pretrained_weights) 

    # get data loaders
    train_loader = get_dataloader(args.batch_size, is_train=True, to_shuffle=True)
    test_loader = get_dataloader(args.batch_size, is_train=False, to_shuffle=True)

	# train and validate
    print('epochs {}'.format(args.epochs + 1))
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval, logger)
        test(model, device, test_loader, epoch, logger)
        if (args.save_model):
            save_model(model, os.path.join(gettempdir(), args.save_name))
        logger.current_logger().report_text('The default output destination for model snapshots and artifacts is: {}'.format(args.save_name))
