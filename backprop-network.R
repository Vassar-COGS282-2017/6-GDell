# source('load-mnist-data.R') # run this once, and then comment out to save processing time.

n.inputs <- 784 # The input images are 28x28, for a total of 784 pixels
n.hidden <- 50 # Number of hidden layer nodes. You can adjust this parameter once you get the network working.
n.output <- 10 # Number of output nodes. Needs to be 10. This is coarse coding the category--each node corresponds to a digit (0-9).
learning.rate <- 0.2# Learning rate parameter for backprop. You can adjust this.

# The full data set has 60,000 training examples and 10,000 test examples. Running the network on all of these
# can take several minutes at a minimum. To speed up development and testing, we'll randomly sample a subset
# of the training and testing examples for each epoch. We'll draw the sample from the full set of examples for 
# each epoch to avoid overfitting the network to a particular set of training examples.

epoch.train.size <- 100 # randomly sample this many training examples (max = 60,000)
epoch.test.size <- 50 # randomly sample this many test examples (max=10,000)

# This is the sigmoid activation function.
sigmoid.activation <- function(x){
  return(1 / (1+exp(-x)))
}

# Add some code to visualize the sigmoid activation function from x=-10 to x=10.

temp <- c()

count <- 1
for(var in -10:10) {
  temp[count] <- sigmoid.activation(var)
  count <- count + 1
}

plot(temp)

# Initializing the matrices to hold the connection weights.
# We'll create one matrix for the weights from the input to hidden layer, and another for the weights from the 
# hidden to output layer. (You could write more effecient code by merging these into the same variable, but
# this approach makes it a little bit easier to understand what is happening in the code).
# The weights are initialized to random numbers drawn from a normal distribution with mean=0 & sd=0.1.
input.to.hidden.weights  <- matrix(rnorm(n.inputs*n.hidden,mean=0,sd=0.1), nrow=n.inputs, ncol=n.hidden)
hidden.to.output.weights <- matrix(rnorm(n.hidden*n.output,mean=0,sd=0.1), nrow=n.hidden, ncol=n.output)

# Write the forward pass function for the network.
# The input parameter will be a vector that is n.inputs long.
# You should first calculate the activation of the hidden layer, and then calculate the activation
# of the output layer. We're going to need both sets of activations in the backprop method, so I've
# written the return() statement to return both hidden.layer and output.layer activations. Your
# job is to create the hidden.layer and output.layer variables in the appropriate way.
# Don't forget to use the sigmoid.activation function created above.
forward.pass <- function(input){
  # input.to.hidden.weights, hidden.to.output.weights
  hidden.layer.activation <- sigmoid.activation(input) * input.to.hidden.weights
  # print(hidden.layer.activation)
  # hidden.layer.activation <- sigmoid.activation(hidden.layer.activation)
  # print(hidden.layer.activation)
  output.layer.activation <- sigmoid.activation(hidden.layer.activation) * as.vector(hidden.to.output.weights)
  # output.layer.activation <- sigmoid.activation(output.layer.activation)
  return(list(hidden=hidden.layer.activation, output=output.layer.activation))
}

# Now we can write the backward pass, or the backprop function.
# The input will be a vector that is n.inputs long
# The output will be a vector that is n.output long
# Backprop is very similar to the delta rule, with a few modifications. In fact, the delta rule is 
# a special case of backprop. As a reminder, this is the Delta rule:
# change.in.weight.from.i.to.j = learning.rate * error.in.j * activation.of.i
# One piece of the delta rule that is commonly left out, and we did this in our example in class, is the 
# derivative of the activation function of node j. We left this out because we used the activation
# function output = x, so the derivative is 1 (The derivative describes the rate of change of output as 
# x changes. When output = x, a change of 1 in x will cause a corresponding change of 1 in output).
# However, now we are using the sigmoid activation function, so we need to include this derivative term again.
# The full delta rule is therefore:
# change.in.weight.from.i.to.j = learning.rate * error.in.j * derivative.of.activation.funtion.for.j.at.current.activation.level * activation.of.i
# This equation will describe all of the weight updates that we do in backprop. The only missing piece is how to get the error
# terms for the hidden layer units. We know the error for the output, but need to back-propogate this error to the hidden layer.
# I'll describe how this works below.

backprop <- function(input, target){
  
  # Step 1. Because we are going to modify the weights in stages, we need to create
  # copies of the weights so that we don't overwrite the original weights before we are
  # done with them. Really, all we need are matrices of the right size to hold the delta
  # (change in) weights, but we can get a matrix of the right size by copying the weight
  # matrices. We'll overwrite all the values before we are done.
  delta.hidden.to.output.weights <- hidden.to.output.weights
  delta.input.to.hidden.weights <- input.to.hidden.weights
  
  # Step 2. Calculate the forward pass on the network. This gets us the activity of the
  # hidden and output layers for a particular set of inputs.
  activations <- forward.pass(input)
  output.activation <- activations$output
  hidden.activation <- activations$hidden
  
  # Step 3. Find the error on the output units. We can find the error just like we did 
  # with the delta rule. Just subtract the actual output from the desired output.
  output.error <- target - output.activation
  # print(paste("error: ", output.error))
  
  # Step 4. We need to multiply the error for a node (an element in the output.error vector)
  # by the derivative of the activation function for the node. The activation function is the
  # sigmoid function: z = 1 / (1 + exp(-x)). The derivative of this function is z * (1 - z).
  # Calculate a "weighted" error in two steps:
  # 1) Find the derivative (slope) for each output node at the level of activation of that node.
  # 2) Multiple this by the node's error to get the weighted error.
  output.slope <- output.activation * (1 - output.activation)
  output.weighted.error <- output.slope * output.error
  
  # Step 5. Find the change in the hidden to output weights by applying the delta rule, using the
  # weighted error instead of the error.
  for(o in 1:n.output){
    delta.hidden.to.output.weights[,o] <- learning.rate * output.weighted.error[o] * output.activation[o]
  }
    # DELTA RULE
   # learning.rate * error.in.j * activation.of.i

  # Step 6. Now we need to "backpropogate" the error from the output nodes to the hidden nodes,
  # so that we can apply the delta rule to the hidden nodes. How much error should we assign to
  # each hidden node? It turns out that the right way to do this is to multiply the error by the
  # connection weight for each node. Each weight will have some error associated with it, and then
  # we sum up all the errors for each hidden node. This makes some intuitive sense. If a hidden node
  # is strongly connected to an output node, then it contributed a lot to the error of that node. If
  # a hidden node has almost no connection (weight near 0), then it contributed very little to the
  # error of the node.
  hidden.error <- output.error * as.vector(hidden.to.output.weights)
  
  # Step 7. Just like the output layer, we need to calculate the weighted error for the hidden nodes.
  hidden.slope <- hidden.activation * (1 - hidden.activation)
  hidden.weighted.error <- hidden.slope * hidden.error
  
  # Step 8. Apply the delta rule using the weighted errors.
  for(h in 1:n.hidden){
    delta.input.to.hidden.weights[,h] <- learning.rate * hidden.weighted.error[h] * hidden.activation[h]
  }
  
  # Step 9. Change the actual weights by the delta amount.
  return(list(
    hidden.to.output.weights=hidden.to.output.weights + delta.hidden.to.output.weights,
    input.to.hidden.weights=input.to.hidden.weights + delta.input.to.hidden.weights
  ))
  
}

# This function takes an input vector and calculates the error of the output
# We can use this to track the progress of the network over time.
test.pattern <- function(input, target){
  activation <- forward.pass(input)
  rmse <- sqrt(mean((target-activation$output)^2))
  return(rmse)
}

# Write a function that returns TRUE if the node with the largest activation on the output layer
# is the same as the single node in the target vector that is equal to 1. Otherwise, it returns
# FALSE. This function represents the best guess of the network. We will use it to interpret the
# output of the network as an identification of the digit.
classification.correct <- function(input, target){

  maxIndexInput <- which.max(input)
  maxIndexTarget <- which.max(target)
  
  if(maxIndexInput == maxIndexTarget) {
    return(TRUE)
  } else {
    return(FALSE)
  }

}

# This function runs a single epoch, based on the epoch.train.size and epoch.test.size parameters
# at the top of this file.
epoch <- function(){
  for(t in sample(60000, epoch.train.size)){
    weights <- backprop(trainData[t,], trainLabels[t,])
    input.to.hidden.weights <- weights$input.to.hidden.weights
    hidden.to.output.weights <- weights$hidden.to.output.weights
  }
  error <- 0
  classification.acc <- 0
  for(t in sample(10000, epoch.test.size)){
    error <- error + test.pattern(testData[t,], testLabels[t,])
    classification.acc <- classification.acc + classification.correct(testData[t,], testLabels[t,])
  }
  classification.acc <- classification.acc / epoch.test.size
  return(list(error=error, classification.accuracy=classification.acc))
}

# Run a batch of epochs
batch <- function(epochs){
  errors <- numeric(epochs)
  classifications <- numeric(epochs)
  pb <- txtProgressBar(min=0, max=epochs, style=3) 
  for(i in 1:epochs){
    result <- epoch()
    errors[i] <- result$error
    classifications[i] <- result$classification.accuracy
    setTxtProgressBar(pb, i)
  }
  return(data.frame(epoch=1:epochs, error=errors, accuracy=classifications))
}

# Uncomment these lines when you are ready to test the code.
result <- batch(300)  # 300 epochs should be enough to reach >80% accuracy.
plot(result$accuracy) # plot the accuracy of the network over training (should increase).
plot(result$error) # plot the error at the output layer over time (should decrease).

# Try adjusting various parameters of the network (number of hidden layer nodes, learning rate) to see if you can improve learning. 

