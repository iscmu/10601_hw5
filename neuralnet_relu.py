"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement ReLU methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming 
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import numpy as np
import argparse
from typing import Callable, List, Tuple
from neuralnet import NN, INIT_FN_TYPE, zero_init, random_init, args2data, Linear, SoftMaxCrossEntropy
# import logging

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')

    
class ReLU:
    def __init__(self):
        """
        Initialize state for ReLU activation layer
        """
        # TODO: Initialize any additional values you may need to store for the
        #  backward pass here
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ReLU activation: f(x) = max(0, x)
        :param x: Input to activation function, shape (output_size,)
        :return: Output of ReLU activation, shape (output_size,)
        """
        # TODO: perform the forward pass for ReLU and save any values you may
        #  need for the backward pass
        # raise NotImplementedError
        self.x = np.maximum(0, x)
        return self.x
    

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        Backward pass through ReLU
        :param dz: partial derivative of loss w.r.t. ReLU output
        :return: partial derivative of loss w.r.t. ReLU input
        """
        # TODO: implement the backward pass
        #  Hint: derivative of ReLU is 1 for positive inputs and 0 for
        #  non-positive inputs (x <= 0)
        # raise NotImplementedError
        return dz * (self.x > 0)

class NN_ReLU(NN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, ReLU) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate

        # TODO: initialize modules (see section 9.1.2 of the writeup)
        #  Hint: use the classes you've implemented above!
        self.linear1 = Linear(self.input_size, self.hidden_size, weight_init_fn, learning_rate)
        self.act1 = ReLU()
        self.linear2 = Linear(self.hidden_size, self.output_size, weight_init_fn, learning_rate)
        self.act2 = SoftMaxCrossEntropy()

if __name__ == "__main__":

    # logfile = '/Users/iskandersergazin/CarngieMellonUniversity/10601/hw5/handout/example.log'
    # loglevel = logging.DEBUG
    # logging.basicConfig(filename=logfile, filemode='a', level=loglevel)
    # logging.info('\n-------------------- START --------------------')

    args = parser.parse_args()
    # Note: You can access arguments like learning rate with args.learning_rate
    # Generally, you can access each argument using the name that was passed 
    # into parser.add_argument() above (see lines 24-44).

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    # Call args2data to get all data + argument values
    # See the docstring of `args2data` for an explanation of 
    # what is being returned.
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)


    nn = NN_ReLU(
        input_size=X_tr.shape[-1],
        hidden_size=n_hid,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=lr
    )

    # train model
    # (this line of code is already written for you)
    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # test model and get predicted labels and errors 
    # (this line of code is written for you)
    train_labels, train_error_rate = nn.test(X_tr, y_tr)
    test_labels, test_error_rate = nn.test(X_test, y_test)

    # Write predicted label and error into file
    # Note that this assumes train_losses and test_losses are lists of floats
    # containing the per-epoch loss values.
    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))