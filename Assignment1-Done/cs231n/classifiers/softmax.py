from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    N_train = X.shape[0] 
    N_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Scores = X.dot(W)

    # Loss
    for i in range(N_train):
        
        temp = Scores[i] - np.max(Scores[i]) # avoiding of overfitting
        exp_temp = np.exp(temp)
        normlized_temp = exp_temp / np.sum(exp_temp)
        loss += -np.log(normlized_temp[y[i]])
        #loss += -temp[y[i]] + np.log(np.sum(exp_temp))
    
        # Grads
        for j in range(N_classes):
        
            dW[:, j] += X[i] * normlized_temp[j]
        dW[:, y[i]] -= X[i]

    
    loss = loss / N_train + reg * np.sum(W * W)
    dW = dW / N_train + 2 * reg * W
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    N_train = X.shape[0] 
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Scores = X @ W # (N, D) x (D, C) = (N, C)
    Scores = Scores - np.max(Scores, axis = 1).reshape(-1, 1) # preventing of overfitting
    exp_scores = np.exp(Scores) 
    sum_scores = np.sum(exp_scores, axis = 1).reshape(-1, 1)
    normalized_scores = exp_scores/sum_scores
    
    corrected_scores = normalized_scores[np.arange(N_train), y]
    
    loss = np.mean(-np.log(corrected_scores)) + 2 * reg * np.sum(W * W)
    
    
    dS = normalized_scores # (N, C)
    dS[np.arange(N_train), y] -= 1
    
    dW = 2 * reg * W +  X.T @ dS/N_train  # (D, N) x (N ,C) = (D, C)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
