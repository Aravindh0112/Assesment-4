# Functions to set up a simple neural network for classification,
# and to train it using stochastic gradient descent.

# Function to initialize a neural network with random weights and biases
netup <- function(d) {
  # Input : d   - vector representing the number of nodes in each layer
  # Output: nn  - a list containing initialized weights, offsets, and node values
  
  nn <- list()
  
  # Initialize h (node values for each layer)
  nn[["h"]] <- lapply(d, function(layer_size) rep(0, layer_size))
  
  # Initialize W (weight matrices)
  nn[["W"]] <- lapply(1:(length(d)-1), function(i) {
    matrix(runif(d[i] * d[i + 1], 0, 0.2), nrow = d[i + 1], ncol = d[i])
  })
  # Initialize b (offset vectors)
  nn[["b"]] <- lapply(1:(length(d)-1), function(i) {
    rep(runif(1, 0, 0.2), d[i + 1])  # Use rep to create a vector of the same bias value for each node
  })
  
  return(nn)
}

# Function to perform a forward pass through the neural network
forward <- function(nn, inp) {
  # Input: nn   - the neural network, 
  #        inp  - vector of input values
  # Output: nn  - the neural network with updated node values
  
  # Update the first layer with input values
  nn[["h"]][[1]] <- inp
  
  # Perform forward pass through the network
  for (l in 2:length(nn[["h"]])) {
    W <- nn[["W"]][[l-1]]
    b <- nn[["b"]][[l-1]]
    h_prev <- nn[["h"]][[l-1]]
    
    # Compute the linear combination followed by ReLU activation
    nn[["h"]][[l]] <- pmax(0, W %*% h_prev + b)
  }
  
  return(nn)
}

# Function to perform a backward pass (back-propagation) through the neural network
backward <- function(nn, k) {
  # Input : nn  - the neural network
  #         k   - index of the output class
  # Output: nn  - the neural network with computed derivatives
  
  L <- length(nn[["h"]])  # Number of layers
  dh <- vector("list", L)  # Derivatives w.r.t. nodes
  dW <- vector("list", L-1)  # Derivatives w.r.t. weights
  db <- vector("list", L-1)  # Derivatives w.r.t. offsets
  
  # Initialize dh for the last layer based on the provided formula
  dh[[L]] <- (exp(nn[["h"]][[L]]) / sum(exp(nn[["h"]][[L]]))) - 
    (1 * (1:length(nn[["h"]][[L]]) == k))
  
  # dh for other nodes using back-propagation through the layers
  for (l in (L-1):1) {
    W <- nn[["W"]][[l]]
    h_next <- nn[["h"]][[l+1]]
    dh[[l]] <- as.vector(t(W) %*% (dh[[l+1]] * (h_next > 0)))
  }
  
  # Compute derivatives w.r.t. weights and offsets
  for (l in 1:(L-1)) {
    h_next <- nn[["h"]][[l+1]]
    h_current <- nn[["h"]][[l]]
    dh_next <- dh[[l+1]]
    dW[[l]] <- (dh_next * (h_next > 0)) %*% t(h_current)
    db[[l]] <- dh_next * (h_next > 0)
  }
  
  # Update the network list with derivatives
  nn[["dh"]] <- dh
  nn[["dW"]] <- dW
  nn[["db"]] <- db
  
  return(nn)
}
