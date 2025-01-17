# Provides the actual implementations for training the model (used by auto_train.py)

import os
from random import randint
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from utils.networks import network_dict
from utils.status import Status
import numpy as np
import torch
from sklearn.metrics import f1_score
from utils.constants import Constants as c
from sklearn.utils import shuffle


class StopTPE(Exception):
    pass


def train_single_run(lr, bs, n_epochs):
    space = {
        'lr' : lr, 
        'bs' : bs, 
        'n_epochs' : n_epochs
    }
    return train(space)

def train(space):
    try :   
        lr = space['lr']
        bs = int(space['bs'])

        train_x = space['train_x']
        train_y = space['train_y']
        val_x = space['val_x']
        val_y = space['val_y']
        status = space['status']
        n_epochs = int(space['n_epochs'])
        results_dir = space['results_dir']
        network_name = space['network_name']
        seed = space['seed']
        criterion = space['criterion']
        model_save_name = space['model_save_name']
        patience = space['patience']
        min_loss_improvement = space['min_loss_improvement']
        
        min_val_loss_run = np.Inf
        train_losses = []
        valid_losses = []

        n_channels = train_x.shape[1]
        input_length = train_x.shape[-1]
        n_outputs = len(np.unique(train_y))

        stopped_by_user = False

        print("\n\n***********************************************")
        print("Start training - BS: {}, LR: {}, seed: {}".format(bs, lr, seed))
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        model = network_dict[network_name](input_length, n_channels, n_outputs)
        train_on_gpu = c.is_gpu_available
        if train_on_gpu:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        batch_size = bs

        patience_counter = 0

        model_saved = False
        
        for epoch in range(1, n_epochs+1):
            train_x, train_y = shuffle(train_x, train_y)
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            model.train()
            torch.cuda.empty_cache()
            for b in range(0,len(train_x),batch_size):
                # clear the gradients of all optimized variables
                if c.enable_speedup:
                    for param in model.parameters():
                        param.grad = None
                else:
                    optimizer.zero_grad()

                inpt = train_x[b:b+batch_size,:]
                target = train_y[b:b+batch_size]

                x_batch = torch.tensor(inpt,dtype=torch.float32)
    #             x_batch = x_batch.view(x_batch.size()[0], 1, x_batch.size()[1]) # reshape so it can be used with 1d conv
                y_batch = torch.tensor(target,dtype=torch.long)
                if train_on_gpu:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                # forward pass: compute predicted outputs by passing inputs to the model
                # outputs from the rnn
                output = model(x_batch)
                loss = criterion(output, y_batch)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                if model.__class__.__name__ == 'LSTM':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * x_batch.size(0)

            ######################    
            # validate the model #
            ######################
            model.eval()
            torch.cuda.empty_cache()
            val_acc = 0

            y_pred = []

            for b in range(0,len(val_x),batch_size):
                inpt = val_x[b:b+batch_size,:]
                target = val_y[b:b+batch_size]

                x_batch = torch.tensor(inpt,dtype=torch.float32)
    #             x_batch = x_batch.view(x_batch.size()[0], 1, x_batch.size()[1]) # reshape so it can be used with 1d conv
                y_batch = torch.tensor(target,dtype=torch.long)
                if train_on_gpu:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                output = model(x_batch)
                # calculate the batch loss
                loss = criterion(output, y_batch)
                # update average validation loss 
                valid_loss += loss.item()*x_batch.size(0)

                output = torch.exp(output) # only when output of network is log softmax
                preds = torch.argmax(output, dim=1)
                equal = y_batch == preds
                val_acc += equal.sum().item()

                y_pred += list(preds.cpu().numpy())

            y_pred = np.array(y_pred)

            f1 = f1_score(val_y, y_pred, average='micro')

            # calculate average losses
            train_loss = train_loss/len(train_x)
            valid_loss = valid_loss/len(val_x)
            val_acc = val_acc/len(val_x)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # print training/validation statistics 
            if epoch % 10 == 0:
                print('  Epoch: {} \tTrain Loss: {:.6f} \tVal Loss: {:.6f} \tVal f1: {:.2f}%'.format(
                    epoch, train_loss, valid_loss, f1*100))
                
            if valid_loss + min_loss_improvement <= min_val_loss_run:
                min_val_loss_run = valid_loss
                patience_counter = 0
                
            # save model if validation loss has decreased
            if valid_loss + min_loss_improvement <= status.best_valid_loss:
                patience_counter = 0
                print('      --> Model saved! - Epoch: {} - BS: {} - LR: {} - Valid loss: {:.6f} --> {:.6f} - Valid F1: {:.2f}% --> {:.2f}%'.format(
                    epoch,
                    bs,
                    lr,
                    status.best_valid_loss,
                    valid_loss,
                    status.best_f1_valid * 100,
                    f1 * 100
                ))

                status.best_valid_loss = valid_loss
                status.best_train_loss = train_loss
                status.best_bs = bs
                status.best_lr = lr
                status.best_seed = seed
                status.best_f1_valid = f1
                status.optimizer = optimizer.__class__.__name__
                status.criterion = criterion.__class__.__name__
                status.n_epochs = n_epochs
                status.patience = patience
                status.min_loss_improvement = min_loss_improvement

                if c.save_only_best_model:
                    torch.save(model.state_dict(), os.path.join(results_dir, model_save_name + '.pt') )
                    model_saved = True
                    status.epoch_model_saved = epoch
                elif f1 > c.store_model_accuracy_threshold:
                    print('found model that is good enough with this seed!')
                    torch.save(model.state_dict(), os.path.join(results_dir, model_save_name + '.pt') )
                    model_saved = True
                    status.epoch_model_saved = epoch
                    break
                    
            patience_counter += 1

            if patience_counter >= patience:
                print('!!!!!!!!! EARLY STOPPING !!!!!!!!!')
                print('Validation loss did not improve at least {} over the last {} epochs'.format(min_loss_improvement, patience))
                break

        status.optimization_losses.append(min_val_loss_run)
        # save the plots and configuration of the best model
        if model_saved:
            status.best_train_losses = train_losses
            status.best_valid_losses = valid_losses

            if not c.save_only_best_model:
                raise StopTPE # exception necessary to break out of TPE hyperparameter search
        
        
        
        return {
            'status' : STATUS_OK,
            'loss' : min_val_loss_run
        }
    except RuntimeError as e:
        print(e)
        return {
            'status' : STATUS_FAIL
        }



def grid_search(status, n_epochs, batch_sizes = [], learning_rates = []):
    status.clear()
    run_losses = []

    for bs in batch_sizes:
        for lr in learning_rates:
            run_loss = train_single_run(lr, bs, n_epochs)['loss']
            run_losses.append(run_loss)
                            
    status.print_summary()



def tpe_search(training_conf):
    '''Performs a hyperparameter search using the tree of parzen estimator
    '''
    status = Status()

    if hasattr(c, 'lr') and (c.lr is not None) and (c.lr > 0):
        learning_rate = c.lr
    else:
        learning_rate = hp.loguniform('lr', np.log(c.lr_min), np.log(c.lr_max))

    if hasattr(c, 'bs') and (c.bs is not None) and (c.bs > 0):
        batch_size = c.bs
    else:
        batch_size = hp.quniform('bs', c.bs_min, c.bs_max, c.bs_steps)

    space = {
        'lr': learning_rate,
        'bs': batch_size, 
        'n_epochs' : c.n_epochs,
        'model_save_name' : c.model_save_name,
        'status' : status,
        'results_dir' : training_conf['results_dir'],
        'network_name': training_conf['network_name'],
        'seed' : training_conf['seed'],
        'criterion': training_conf['criterion'],
        'train_x': training_conf['train_x'],
        'train_y': training_conf['train_y'],
        'val_x': training_conf['val_x'],
        'val_y': training_conf['val_y'],
        'patience': c.patience,
        'min_loss_improvement': c.min_loss_improvement
    }

    try:
        best = fmin(
            fn=train,
            space=space,
            algo=tpe.suggest,
            max_evals=c.n_tries
        )
    except KeyboardInterrupt:
        pass
    except StopTPE:
        print('controlled tpe abort!')

    return status


def evaluate_model(model, criterion, test_x, test_y):
    '''Evaluates a model, provided the loss function and test data
    Args:
        model: Model that should be evaluated
        criterion: Loss function
        test_x: Test data
        test_y: Test labels
    Returns:
        test_loss (float): Average test loss
        test_acc (float): Average accuracy 
        f1 (float): F1 score
    '''
    model.eval()

    train_on_gpu = c.is_gpu_available

    if train_on_gpu:
        model = model.cuda()

    test_acc = 0
    batch_size = 128
    y_pred = []
    test_loss = 0.0

    for b in range(0,len(test_x),batch_size):
        inpt = test_x[b:b+batch_size,:]
        target = test_y[b:b+batch_size]

        x_batch = torch.tensor(inpt,dtype=torch.float32)
        y_batch = torch.tensor(target,dtype=torch.long)
        if train_on_gpu:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        output = model(x_batch)
        # calculate the batch loss
        loss = criterion(output, y_batch)
        # update average validation loss 
        test_loss += loss.item()*x_batch.size(0)

        output = torch.exp(output) # only when output of network is log softmax
        preds = torch.argmax(output, dim=1)
        equal = y_batch == preds
        test_acc += equal.sum().item()

        y_pred += list(preds.cpu().numpy())

    y_pred = np.array(y_pred)

    f1 = f1_score(test_y, y_pred, average='micro')

    # calculate average losses
    test_loss = test_loss/len(test_x)
    test_acc = test_acc/len(test_x)

    return test_loss, test_acc, f1