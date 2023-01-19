from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from operators import train, predict
from data.mnist import load_mnist

def main(train_data, train_labels, config):
    # train (5,000 data)
    history, parameters = train(train_data, train_labels, **config)

    # prediction
    prediction = predict(test_data, parameters, config.activation_fn)

    # Visualize
    # loss
    train_loss = history['train_loss']
    valid_loss = history['valid_loss']

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    x_len = np.arange(1, len(train_loss)+1)
    plt.plot(x_len, train_loss, marker='.', c='blue', label="train_loss")
    plt.plot(x_len, valid_loss, marker='.', c='red', label="valid_loss")

    plt.title('Train_Validation Loss')
    plt.legend() # loc='upper right'
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # evaluation
    metric = ''.join(history['evaluation'][0].keys())
    score = [history['evaluation'][i][metric]['score'] for i in range(len(history['evaluation']))]

    plt.subplot(1, 2, 2)
    plt.plot(x_len, score, marker='.', c='green', label=metric)

    plt.title('Evaluation Score')
    plt.legend() # loc='upper right'
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.show()

if __name__ == "__main__":

    # Load MNIST datasets
    train_set, test_set = load_mnist()
    train_data, train_labels = train_set
    test_data, test_labels = test_set

    # normalize
    train_data = train_data / 255
    test_data = test_data / 255

    # load config
    config = OmegaConf.load('config.yaml')
    # fix data if you need like...
    # config.epoch = 5, config.batch_size = 64

    # show details
    print(f"Config details:\n{'='*20}\n\n{OmegaConf.to_yaml(config)}{'='*20}", end="\n\n")

    # run main
    main(train_data, train_labels, config)