
###########################################################################

netup <- function(d) {
  
  neural_net <- list()
  print(d)
  print(length(d))
  
  neural_net$h <- lapply(d , function(layer_sizeee) numeric(layer_sizeee))
  
  neural_net$W <- lapply(2:length(d) , function(layer_length) {
    matrix(runif(d[layer_length - 1] * d[layer_length] , 0, 0.2) , nrow = d[layer_length - 1])
    })
  
  neural_net$b <- lapply( 2:length(d) , function(layer_length) {
    runif(d[layer_length] , 0 ,0.2)
  })
  
  return (neural_net)
  
}


neural_net <- netup(c(3,4,4,2))

print(dim(neural_net$h))
print(neural_net$h)
print(neural_net$W)
print(neural_net$b)


#################################################################

forward <- function (nn , inp) {
  
  hid_layer <- nn$h
  hid_layer[[1]] <- inp
  
  print(dim(matrix(nn$W[[2]])))
  
  hid_layer <- lapply( 2 : length(nn$h) , function(layer) {
    
  layer_value <- ( matrix(nn$W[[layer-1]]) %*% matrix(nn$h[[layer-1]]) + 
     matrix(nn$b[[layer - 1]], ncol = ncol(nn$W[[layer - 1]]), byrow = TRUE))
  
  pmax(0,layer_value)
    
  })
  
  return (nn)

}

network <- netup(c(3,4,4,2))

input_vector <- c(3,4,5)

print(network)

updated_neural_net <- forward(network , input_vector)

print(updated_neural_net$h)
print(updated_neural_net$W)
print(updated_neural_net$b)

##################################################################
