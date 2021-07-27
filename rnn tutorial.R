#
#https://www.kaggle.com/rtatman/beginner-s-intro-to-rnn-s-in-r
library(keras) # for deep learning
library(tidyverse) # general utility functions
library(caret) # machine learning utility functions

weather_data <- read_csv("seattleWeather_1948-2017.csv")
head(weather_data)

# set some parameters for our model
max_len <- 6 # the number of previous examples we'll look at
batch_size <- 32 # number of sequences to look at at one time during training
total_epochs <- 15 # how many times we'll look @ the whole dataset while training our model

# set a random seed for reproducability
set.seed(123)
?na.omit
?layer_dense
rain <- weather_data$RAIN
table(rain)

# Our next task is to take this vector and then chop it up into samples of our max_length + 1 (since we want to know what the next item in the sequence is in order to see how good we did in predicting it: we'll peel it off before we pass the training data to our model). We could just start at the beginning of our vector and chop it into non-overlapping chunks of max_length + 1, but that would only give us around 3600 total examples, which really isn't enough to train a deep learning model. (To be honest, this problem is probably too small/simple to use deep learning for at all. But this tutorial is about learning how to build an RNN, rather than picking the best possible model for this problem, we're going to pretend it's not.)
                                                  # To stretch out our data, we can use something called moving-block sub-sampling, which is where we cut up our vector into overlapping chunks. Like so:

# In the figure above, the blocks are three units long and overlap by two units. We're going to have blocks that are max_length + 1 long and overlap by 3.

# Cut the text in overlapping sample sequences of max_len characters

# get a list of start indexes for our (overlapping) chunks
start_indexes <- seq(1, length(rain) - (max_len + 1), by = 3)

# > head(start_indexes)
# [1]  1  4  7 10 13 16
# > tail(start_indexes)
# [1] 25528 25531 25534 25537 25540 25543

# create an empty matrix to store our data in
weather_matrix <- matrix(nrow = length(start_indexes), ncol = max_len + 1)

dim(weather_matrix)
# [1] 8515    7  #although we started with 25,551 days of rain, we are dividing them into blocks of 7. 25551/7 = 3650, but we created overlapping intervals and so we now have 8515 examples (inputs) with a window length of 7 (6 plus the class)

#now we have to fill the matrix with our rain data. Note this is a boolean vector
rain[2:4]
#[1] TRUE TRUE TRUE

length(start_indexes)
#8515
# we loop through each element of the start_indexes  -8515 of them. We fill in the rows of the matrix with rain[currentStartIndex: currentStartIndex +7]
#which can also be written using the loop index i
#as 
#rain[start_indexes[i]: start_indexes[i] +7]
#the current row of the matrix is i and so:
#weather_matrix[i,] <- rain[currentStartIndex: currentStartIndex +max_len]

# fill our matrix with the overlapping slices of our dataset
for (i in 1:length(start_indexes)){
  weather_matrix[i,] <- rain[start_indexes[i]:(start_indexes[i] + max_len)]
}

# Now that we have our sample data in a tidy matrix, we just need to do a couple of housekeeping things to make sure everything is good to go.
# 
# Make sure your matrix is numeric. Since Keras expects a numeric matrix, we're going to convert our data from boolean to numeric by multiplying everything by one. This won't change any values, but will change them to numeric datatype and throw an error if you've happened to insert a text string
# Remore any na's: If you accidentally end up with na's in your data, your model will compile and train just fine... but your model predictions will all be NaN (not a number). I like to always include this step just in case, but in this case we do need to do it since, as is shown in the diagram above, our data slicing approach ended up adding na's to the dataset.


# make sure it's numeric
weather_matrix <- weather_matrix * 1
head(weather_matrix)
# [,1] [,2] [,3] [,4] [,5] [,6] [,7]
# [1,]    1    1    1    1    1    1    1
# [2,]    1    1    1    1    1    1    1
# [3,]    1    1    1    1    1    0    0
# [4,]    1    1    0    0    0    0    0
# [5,]    0    0    0    0    0    0    0
# [6,]    0    0    0    0    0    0    1

# ?na.omit
# na.omit removes rows with NA in them.
DF <- data.frame(x = c(1, 2, 3), y = c(0, 10, NA))
na.omit(DF)
# x  y
# 1 1  0
# 2 2 10

anyNA(weather_matrix)
# [1] TRUE

# remove na's if you have them
if(anyNA(weather_matrix)){
  weather_matrix <- na.omit(weather_matrix)
}

# Alright, now that our data is all clean and ready to go, we can get down to preparing it to be fed into our model. 

#we take the first 6 elements (first max_len-1 elements) and put them in a matrix we call X and take the last elements and put them in a matrix called y.

# First, we're going to need to split our dataset into the input (the six previous days) and the output (the single day we're interested in predicting). I'm going to follow convention and call my inputs X and my output y.

# split our data into the day we're predict (y), and the 
# sequence of days leading up to it (X)
X <- weather_matrix[,-ncol(weather_matrix)]
y <- weather_matrix[,ncol(weather_matrix)] #note that R behaves a bit strangely with matrices with one column. It treats them as vectors.


# Now we just need to split our data in to the testing and training set using the createDataPartition() function from the caret package. From this point on, we're not going to even think about looking at our testing dataset until it's time to evaluate our final model.


# create an index to split our data into testing & training sets
training_index <- createDataPartition(y, p = .9, 
                                      list = FALSE, 
                                      times = 1)

# training data
X_train <- array(X[training_index,], dim = c(length(training_index), max_len, 1))
y_train <- y[training_index]

# testing data
X_test <- array(X[-training_index,], dim = c(length(y) - length(training_index), max_len, 1))
y_test <- y[-training_index]











