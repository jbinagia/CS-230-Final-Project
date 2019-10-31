"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch import distributions
from torch import nn

import project.utils as utils
import project.networks.net as net
import project.networks.data_loader as data_loader
from project.evaluation.evaluate import evaluate

# the following code was previously used when this script could be ran standalone
# parser = argparse.ArgumentParser() # create a parser for command line arguments
# parser.add_argument('--data_dir', default='data/crescent',
#                     help="Directory containing the dataset")
# parser.add_argument('--model_dir', default='experiments/base_model',
#                     help="Directory containing params.json")
# parser.add_argument('--restore_file', default=None,
#                     help="Optional, name of the file in --model_dir containing weights to reload before \
#                     training")  # 'best' or 'train'


def train_model(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t: # creates both progress bars
        for i, (train_batch, labels_batch) in enumerate(dataloader): # for the i-th batch
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = net.realnvp_loss_fn(output_batch, model) # for realNVP
            # loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            #loss.backward()
            loss.backward(retain_graph=True)

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg())) # adds avg loss to second progress bar
            t.update() # updates the progress bar

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None, torch_save_file = 'testing'):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_model(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(model.state_dict(), 'results/training/' + torch_save_file + '.pt')


# if __name__ == '__main__':
def train(model = None, model_dir = 'project/experiments/base_model/', \
    data_dir = 'project/data/crescent/', torch_save_file = 'testing', restore_file = None):

    # define paths
    full_model_path = os.path.join(os.path.dirname(os.getcwd()),model_dir)
    full_data_path = os.path.join(os.path.dirname(os.getcwd()),data_dir)
    full_restore_path = (os.path.join(os.path.dirname(os.getcwd()),restore_file) if restore_file is not None else None)

    # Load the parameters from json file
    # args = parser.parse_args() # inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action.
    # json_path = os.path.join(args.model_dir, 'params.json') # json file storing parameters like learning rate
    json_path = full_model_path + 'params.json' # json file storing parameters like learning rate
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path) # defined in utils.py. Class that loads hyperparameters from a json file.

    # use GPU if available
    params.cuda = torch.cuda.is_available() # torch.cuda.is_available() returns a bool indicating if CUDA is currently available.

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    #utils.set_logger(os.path.join(args.model_dir, 'train.log')) # create logger that saves every output to the terminal in a permanent file
    utils.set_logger(full_model_path + 'train.log') # create logger that saves every output to the terminal in a permanent file

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    # dataloaders = data_loader.fetch_dataloader(
    #     ['train', 'val'], args.data_dir, params)
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], full_data_path, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    logging.info("- done.")

    # Define the model and optimizer
    if model == None: # then use default as defined here
        nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh()) # net s
        nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2)) # net t
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32)) # 6x2 matrix. len(masks) = 6 = num subblocks.
        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))      # so we have a total of 3 neural blocks (see fig. 1 of boltzmann generators paper)
        model = net.RealNVP(nets, nett, masks, prior).cuda() if params.cuda else net.RealNVP(nets, nett, masks, prior)
    else:
        model = model.cuda() if params.cuda else model # send to gpu if possivble
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=params.learning_rate)

    # model = net.Net(params).cuda() if params.cuda else net.Net(params) # Calling .cuda() on a model/Tensor/Variable sends it to the GPU
    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate) # we want to exclude the mask from the update step

    # fetch loss function and metrics
    # loss_fn = net.loss_fn
    loss_fn = net.realnvp_loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    # train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
    #                    args.restore_file)
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, full_model_path,
                       full_restore_path, torch_save_file)
