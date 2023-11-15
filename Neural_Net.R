##################################################################

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

neural_net <- netup(c(3,4,4,2))

print(neural_net$h)
print(neural_net$W)
print(neural_net$b)

###############################################################

forward <- function(nn, inp) {
  # Input: nn   - the neural network, 
  #        inp  - vector of input values
  # Output: nn  - the neural network with updated node values
  
  # Update the first layer with input values
  nn[["h"]][[1]] <- inp
  
  # Perform forward pass through the network
  for (l in 2:length(nn[["h"]])) {
    print(l)
    W <- nn[["W"]][[l-1]]
    b <- nn[["b"]][[l-1]]
    h_prev <- nn[["h"]][[l-1]]
    
    # Compute the linear combination followed by ReLU activation
    nn[["h"]][[l]] <- pmax(0, W %*% h_prev + b)
  }
  
  return(nn)
}


network <- netup(c(3,4,4,2))

input_vector <- c(3,4,5)

print(network)

updated_neural_net <- forward(network , input_vector)

print(updated_neural_net$h)
print(updated_neural_net$W)
print(updated_neural_net$b)

###################################################################