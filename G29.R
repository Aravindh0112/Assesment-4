# Function to initialize a neural network with random weights and offsets
netup <- function(d) {
  # Input : d   - vector representing the number of nodes in each layer
  # Output: nn  - a list containing initialized weights, offsets, and node values
  
  nn <- list(
    # Initialize h (node values for each layer)
    h = lapply(d, function(layer_size) rep(0, layer_size)),
    # Initialize W (weight matrices)
    W = lapply(1:(length(d)-1), function(i) matrix(runif(d[i] * d[i + 1], 0, 0.2), 
                                                   nrow = d[i + 1], ncol = d[i])),
    # Initialize b (offset vectors)
    b = lapply(1:(length(d)-1), function(i) rep(runif(1, 0, 0.2), d[i + 1]))
  )
  return(nn)
}

# Function to perform a forward pass through the neural network
forward <- function(nn, inp) {
  # Input : nn  - the neural network, 
  #         inp - vector of input values
  # Output: nn  - the neural network with updated node values
  
  # Update the first layer with input values
  nn$h[[1]] <- inp
  
  # Perform forward pass through the network
  for (l in 2:length(nn$h)) {
    W <- nn$W[[l-1]]
    b <- nn$b[[l-1]]
    h_prev <- nn$h[[l-1]]
    
    # Compute the linear combination followed by ReLU activation
    # Formula : h_j^(l+1) = max(0, W_j^l * h^l + b^l)
    # Where   : h_j^(l+1) = 'j'th node value for layer (l+1)
    #         : W_j^1     = 'j'th row of the weight parameter matrix W^l 
    #                        linking layer l to layer (l+1)
    #         : h^l       = vector of node values for layer l
    #         : b^l       = vector of offset parameters linking layer l to layer l + 1
    nn$h[[l]] <- pmax(0, W %*% h_prev + b)
  }
  
  return(nn)
}
