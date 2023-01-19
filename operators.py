import numpy as np
from tqdm import tqdm
from utils import Computations
from utils import Initializers, Optimizers
from dataloader import train_validation_split, split_into_batches

def train(datasets, labels, model, val_size, epoch, batch_size, seed, 
          shuffle, init_fn, activation_fn:list, loss_fn, optim, optim_args, eval_metric):
    """Train & validation
    """
    
    # init parameters
    initializers = Initializers(init_fn)
    parameters = initializers.initialize(model, seed=seed)
    optimizer = Optimizers(optim)(**optim_args)
    
    # train validation split
    train_X, train_Y, val_X, val_Y = train_validation_split(datasets, labels, val_size)
    
    # crawling loss and evaluation
    train_loss_list = []
    validation_loss_list = []
    evaluation_list = []
    best_eval_score = 0.0
    for i in range(epoch):
        
        # split train data by batch size
        train_X_datasets, train_Y_datasets = split_into_batches(train_X, train_Y, batch_size=batch_size, drop=True, seed=seed, shuffle=shuffle)
        
        # train iteration
        for batches_X, batches_Y in zip(tqdm(train_X_datasets, leave=False), train_Y_datasets):
            
            # forward
            prediction, cache = Computations.feed_forward(batches_X, parameters, activation_fn)
            # backward
            gradients = Computations.back_prop(prediction, batches_Y, cache, parameters, activation_fn, loss_fn)
            # update
            parameters = optimizer.update(parameters, gradients)
        
        # validate per epoch
        # train_loss check
        train_loss_list.append(Computations.compute_loss(train_X, train_Y, parameters, activation_fn, loss_fn)) # 예측연산이 내장되어 있음
        # validation loss check
        validation_loss_list.append(Computations.compute_loss(val_X, val_Y, parameters, activation_fn, loss_fn))
        # evaluation
        eval_result = Computations.compute_evaluation(val_X, val_Y, parameters, activation_fn, eval_metric)
        evaluation_list.append(eval_result)
    
        
        # update best eval score
        curr_eval_score = eval_result[eval_metric]['score']
        best_eval_score = curr_eval_score if curr_eval_score > best_eval_score else best_eval_score
        
        
        # verbose
        print(
            f"{str(i+1).zfill(3)} Epoch | \
train_loss: {train_loss_list[-1]:.6f} | \
val_loss: {validation_loss_list[-1]:.6f} | \
val_score: {evaluation_list[-1][eval_metric]['score']:.6f}"
        )

    print(f"Best score: {best_eval_score:.6f}")
    
    history = {
        "train_loss": train_loss_list, 
        "valid_loss": validation_loss_list, 
        "evaluation": evaluation_list,
        "best_score": best_eval_score
    }

    return history, parameters


def predict(datasets, parameters, activation_fn):
    """Prediction
    """
    
    prediction, _ = Computations.feed_forward(datasets, parameters, activation_fn)
    prediction_indices = np.argmax(prediction, axis=1).tolist()
    
    return prediction_indices