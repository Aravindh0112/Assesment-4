# Aravindh Sankar Ravisankar : s2596860
# Md.Nayem Dewan : s2613404
# Aman Syed : s2496727

# Github link: https://github.com/Aravindh0112/G29_Assesment-4.git

# Contributions : This assignment is completed with combined effort of all
# team members. Aravindh developed the netup and forward function , where
# from thereon Nayem coded on the backward and train functions and atlast,
# on incorporating the above functions , Aman worked on finding the optimal 
# seed to minimize loss and fetched the misclassification rate on the test set.

# Set of functions to set up a simple neural network for classification,
# and to train it using stochastic gradient descent , thereby
# predicting on the test data and finding the misclassification rate.

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

# Function to perform a backward pass (back-propagation) through the neural network
backward <- function(nn, k) {
  # Input : nn  - the neural network with updated node values.
  #         k   - index of the output class
  # Output: nn  - the neural network with computed derivatives
  
  L <- length(nn$h) # Number of nodes layers
  dh <- vector("list", L) # Initialize dh to store derivatives of the loss for k w.r.t. nodes
  dW <- vector("list", L-1) # Initialize dW to store derivatives of the loss for k w.r.t. weights
  db <- vector("list", L-1) # Initialize db to store derivatives of the loss for k w.r.t. offsets
  
  # Initialize dh for the last layer based on the provided formula
  # Formula : dh_j^L = exp(h_j^L) / sum(exp(h^L)) if j!= k
  #         : dh_j^L = (exp(h_j^L) / sum(exp(h^L))) - 1 if j == k
  # Where   : dh_j^L = derivative of the loss for k w.r.t. 'j'th node value of last layer
  #         : h_j^L  = 'j'th node value of last layer
  #         : h^L    = vector of all node values of last layer
  dh[[L]] <- (exp(nn$h[[L]]) / sum(exp(nn$h[[L]]))) - (1 * (1:length(nn$h[[L]]) == k))
  
  # dh for other nodes using back-propagation through the layers
  # Formula : dh^l       = transpose(W^l) * d^(l+1)
  # Where   : dh^l       = derivatives of the loss for k w.r.t. 'l'th layer nodes
  #         : W^l        = weight parameter matrix W^l linking layer l to layer (l+1)
  #         : d^(l+1)    : d_j^(l+1) = dh_j^(l+1) if h_j^(l+1) > 0, else 0
  #         : dh_j^(l+1) = derivative of the loss for k w.r.t. 'j'th node value of (l+1)th layer
  #         : h_j^(l+1)  = 'j'th node value for layer (l+1)
  for (l in (L-1):1) {
    W <- nn$W[[l]]
    h_next <- nn$h[[l+1]]
    dh[[l]] <- as.vector(t(W) %*% (dh[[l+1]] * (h_next > 0)))
  }
  
  # Compute derivatives w.r.t. weights and offsets
  # Formula : db^l       = d^(l+1)
  # Formula : dW^l       = d^(l+1) * transpose(h^l)
  # Where   : db^l       = derivatives of the loss for k w.r.t. 'l'th layer offset
  #         : dW^l       = derivatives of the loss for k w.r.t. 'l'th layer weights
  #         : d^(l+1)    : d_j^(l+1) = dh_j^(l+1) if h_j^(l+1) > 0, else 0
  #         : dh_j^(l+1) = derivative of the loss for k w.r.t. 'j'th node value of (l+1)th layer
  #         : h_j^(l+1)  = 'j'th node value for layer (l+1)
  #         : h^l        = vector of all node values of 'l' th layer
  for (l in 1:(L-1)) {
    h_next <- nn$h[[l+1]]
    h_current <- nn$h[[l]]
    dh_next <- dh[[l+1]]
    dW[[l]] <- (dh_next * (h_next > 0)) %*% t(h_current)
    db[[l]] <- dh_next * (h_next > 0)
  }
  
  # Update the network list with derivatives
  nn$dh <- dh
  nn$dW <- dW
  nn$db <- db
  
  return(nn)
}

# Function to train the neural network using stochastic gradient descent
train <- function(nn, inp, k, eta = 0.01, mb = 10, nstep = 10000) {
  # Input : nn    - the updated neural network with computed derivatives, 
  #         inp   - input data, 
  #         k     - target classes,
  #         eta   - step size,
  #         mb    - mini-batch size, 
  #         nstep - number of steps for optimization
  # Output: nn    - the trained neural network
  
  # Iterate nstep times for optimization
  for (step in 1:nstep) {
    # Randomly sample mini-batch
    indices <- sample(1:nrow(inp), mb, replace = TRUE)
    inp_mb <- inp[indices, , drop = FALSE]
    # target classes for mini-batch
    k_mb <- k[indices]
    
    # Apply forward pass and backward pass 
    # to get dW and db for each value of mini-batch
    dW_ls <- vector("list", mb)
    db_ls <- vector("list", mb)
    for (i in 1:mb) {
      nn_i <- forward(nn, inp_mb[i,])
      nn_i <- backward(nn_i, k_mb[i])
      dW_ls[[i]] <- nn_i$dW
      db_ls[[i]] <- nn_i$db
    }
    
    # Take average of dW
    nn$dW <- lapply(seq_along(dW_ls[[1]]), function(i){
      Reduce(`+`, lapply(dW_ls, function(lst) lst[[i]])) / length(dW_ls)
    })
    # Take average of db
    nn$db <- lapply(seq_along(db_ls[[1]]), function(i){
      Reduce(`+`, lapply(db_ls, function(lst) lst[[i]])) / length(db_ls)
    })
    
    # Update parameters (W and b) using stochastic gradient descent
    # Formula : W^l  ← W^l − eta * dW^l
    # Formula : b^l  ← b^l − eta * db^l
    # Where   : eta  = step size
    #         : dW^l = derivatives of the loss for k w.r.t. 'l'th layer weights
    #         : db^l = derivatives of the loss for k w.r.t. 'l'th layer offsets
    for (l in 1:length(nn$W)) {
      nn$W[[l]] <- nn$W[[l]] - eta * nn$dW[[l]]
      nn$b[[l]] <- nn$b[[l]] - eta * nn$db[[l]]
    }
    # After 1st iteration and thereafter on every 1000th iteration calculate and print loss value
    if(step == 1 | step %% 1000 == 0) {
      # Initialize pki to store the probability that the output variable is in class ki
      pki <- numeric(nrow(inp))
      # iterate over each training input value
      for(i in 1:nrow(inp)){
        # Perform a forward pass to obtain the network's output
        temp_nn <- forward(nn, inp[i,])
        n_layer <- length(nn$h)
        # Extract the output values from the last layer
        h_last_layer <- temp_nn$h[[n_layer]]
        # Calculate pki
        # Formula : pki    = exp(h_ki^L) / sum(h^L)
        # Where   : pki    = the probability that the output variable is in class k for 'i'th input
        #         : h_ki^L = 'k'th nodes value of Last layer for 'i'th input
        #         : h^L    = vector of all node values of Last layer for 'i'th input 
        pki <- exp(h_last_layer[k[[i]]]) / sum(exp(h_last_layer))
      }
      # Calculate loss
      # Formula : Loss = - sum(log(pki)) / n
      # Where   : Loss = Loss
      #         : pki  = the probability that the output variable is in class k for 'i'th input
      #         : n    = number of input values
      loss_value <-  - sum(log(pki)) / nrow(inp)
      # Print loss
      cat('\n The loss value after ',step,' iterations are ', loss_value)
    }
  }
  
  return(nn)
}

# Set seed
set.seed(24)

# Load iris dataset
data(iris)

# Extract features and labels
features <- as.matrix(iris[, 1:4])
labels <- as.numeric(iris$Species)

# Create training and test sets
# Where the test data consists of every 5th row of
# the iris dataset, starting from row 5.
test_indices <- seq(5, nrow(iris), by = 5)

# train dataset has 120 rows
train_features <- features[-test_indices, ]
train_labels <- labels[-test_indices]

#test dataset has 30 rows
test_features <- features[test_indices, ]
test_labels <- labels[test_indices]

# Define neural network architecture (4-8-7-3)
nn_int <- netup(c(4, 8, 7, 3))

# Train the neural network
nn <- train(nn_int, train_features, train_labels, eta = 0.01, mb = 10, nstep = 10000)

# Function to predict using the neural network
predict_nn <- function(nn, features){
  # Input : nn               - the trained neural network
  #         features         - input features for prediction (a matrix where each row is a sample)
  # Output: predicted_labels - vector of predicted labels for each input sample
  
  predicted_labels <- numeric(nrow(features))
  
  # Iterate through each input sample
  for(i in 1:nrow(features)){
    # Perform a forward pass to obtain the network's output
    temp_nn <- forward(nn, features[i,])
    n_layer <- length(nn$h)
    # Extract the output values from the last layer
    h_last_layer <- temp_nn$h[[n_layer]]
    # Predict the label based on the output values
    # Formula : pk    = exp(h_k^L) / sum(h^L)
    # Where   : pk    = the probability that the output variable is in class k
    #         : h_k^L = 'k'th nodes value of Last layer
    #         : h^L   = vector of all node values of Last layer
    predicted_labels[i] <- which.max(exp(h_last_layer) / sum(exp(h_last_layer)))
  }
  
  return(predicted_labels)
}

# Predict the labels for Test Set
predicted_labels <- predict_nn(nn, test_features)

# Compute misclassification rate for Test set
misclassification_rate <- mean(predicted_labels != test_labels)

# Print misclassification rate for Test set
cat("\n Misclassification rate for the test set : ", round(misclassification_rate * 100, 2), "%")

