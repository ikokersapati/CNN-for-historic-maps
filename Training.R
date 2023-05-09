# Convolutional Neural Networks

# Install EBImage
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")

# Install TensorFlow and Keras

install.packages("keras")
library(keras)
install_keras(envname = "r-reticulate")

# Check the installation

library(tensorflow)
tf$constant("Hello Tensorflow!")

# Load packages
library(keras)
library(EBImage)

# Read Images
setwd('C:/Users/user/Downloads/training2/map1')
pic1 <- c('b1.jpg', 'b2.jpg', 'b3.jpg', 'b4.jpg', 'b5.jpg', 'b6.jpg', 'b7.jpg', 'b8.jpg', 'b9.jpg', 'b10.jpg', 'b11.jpg', 'b12.jpg', 'b13.jpg', 'b14.jpg', 'b15.jpg', 'b16.jpg', 'b17.jpg', 'b18.jpg', 'b19.jpg', 'b20.jpg', 'b21.jpg', 'b22.jpg', 'b23.jpg', 'b24.jpg', 'b25.jpg', 'b26.jpg', 'b27.jpg', 'b28.jpg', 'b29.jpg', 'b30.jpg', 'b31.jpg', 'b32.jpg', 'b33.jpg', 'b34.jpg', 'b35.jpg', 'b36.jpg', 'b37.jpg', 'b38.jpg', 'b39.jpg', 'b40.jpg', 'b41.jpg', 'b42.jpg', 'b43.jpg', 'b44.jpg', 'b45.jpg', 'b46.jpg', 'b47.jpg', 'b48.jpg', 'b49.jpg', 'b50.jpg', 'b51.jpg', 'b52.jpg', 'b53.jpg', 'b54.jpg', 'b55.jpg', 'b56.jpg', 'b57.jpg', 'b58.jpg', 'b59.jpg', 'b60.jpg', 'b61.jpg', 'b62.jpg', 'b63.jpg', 'b64.jpg', 'b65.jpg', 'b66.jpg', 'b67.jpg', 'b68.jpg', 'b69.jpg', 'b70.jpg', 'b71.jpg', 'b72.jpg', 'b73.jpg', 'b74.jpg', 'b75.jpg', 'b76.jpg', 'b77.jpg', 'b78.jpg', 'b79.jpg', 'b80.jpg', 'b81.jpg', 'b82.jpg', 'b83.jpg', 'b84.jpg', 'b85.jpg', 'b86.jpg', 'b87.jpg', 'b88.jpg', 'b89.jpg', 'b90.jpg', 'b91.jpg', 'b92.jpg', 'b93.jpg', 'b94.jpg', 'b95.jpg', 'b96.jpg', 'b97.jpg', 'b98.jpg', 'b99.jpg', 'b100.jpg', 'b101.jpg', 'b102.jpg', 'b103.jpg', 'b104.jpg', 'b105.jpg', 'b106.jpg', 'b107.jpg', 'b108.jpg', 'b109.jpg', 'b110.jpg', 'b111.jpg', 'b112.jpg', 'b113.jpg', 'b114.jpg', 'b115.jpg', 'b116.jpg', 'b117.jpg', 'b118.jpg', 'b119.jpg', 'b120.jpg', 'c1.jpg', 'c2.jpg', 'c3.jpg', 'c4.jpg', 'c5.jpg', 'c6.jpg', 'c7.jpg', 'c8.jpg', 'c9.jpg', 'c10.jpg', 'c11.jpg', 'c12.jpg', 'c13.jpg', 'c14.jpg', 'c15.jpg', 'c16.jpg', 'c17.jpg', 'c18.jpg', 'c19.jpg', 'c20.jpg', 'c21.jpg', 'c22.jpg', 'c23.jpg', 'c24.jpg', 'c25.jpg', 'c26.jpg', 'c27.jpg', 'c28.jpg', 'c29.jpg', 'c30.jpg', 'c31.jpg', 'c32.jpg', 'c33.jpg', 'c34.jpg', 'c35.jpg', 'c36.jpg', 'c37.jpg', 'c38.jpg', 'c39.jpg', 'c40.jpg', 'c41.jpg', 'c42.jpg', 'c43.jpg', 'c44.jpg', 'c45.jpg', 'c46.jpg', 'c47.jpg', 'c48.jpg', 'c49.jpg', 'c50.jpg', 'c51.jpg', 'c52.jpg', 'c53.jpg', 'c54.jpg', 'c55.jpg', 'c56.jpg', 'c57.jpg', 'c58.jpg', 'c59.jpg', 'c60.jpg', 'c61.jpg', 'c62.jpg', 'c63.jpg', 'c64.jpg', 'c65.jpg', 'c66.jpg', 'c67.jpg', 'c68.jpg', 'c69.jpg', 'c70.jpg', 'c71.jpg', 'c72.jpg', 'c73.jpg', 'c74.jpg', 'c75.jpg', 'c76.jpg', 'c77.jpg', 'c78.jpg', 'c79.jpg', 'c80.jpg', 'c81.jpg', 'c82.jpg', 'c83.jpg', 'c84.jpg', 'c85.jpg', 'c86.jpg', 'c87.jpg', 'c88.jpg', 'c89.jpg', 'c90.jpg', 'c91.jpg', 'c92.jpg', 'c93.jpg', 'c94.jpg', 'c95.jpg', 'c96.jpg', 'c97.jpg', 'c98.jpg', 'c99.jpg', 'c100.jpg', 'c101.jpg', 'c102.jpg', 'c103.jpg', 'c104.jpg', 'c105.jpg', 'c106.jpg', 'c107.jpg', 'c108.jpg', 'c109.jpg', 'c110.jpg', 'c111.jpg', 'c112.jpg', 'c113.jpg', 'c114.jpg', 'c115.jpg', 'c116.jpg', 'c117.jpg', 'c118.jpg', 'c119.jpg', 'c120.jpg', 'f1.jpg', 'f2.jpg', 'f3.jpg', 'f4.jpg', 'f5.jpg', 'f6.jpg', 'f7.jpg', 'f8.jpg', 'f9.jpg', 'f10.jpg', 'f11.jpg', 'f12.jpg', 'f13.jpg', 'f14.jpg', 'f15.jpg', 'f16.jpg', 'f17.jpg', 'f18.jpg', 'f19.jpg', 'f20.jpg', 'f21.jpg', 'f22.jpg', 'f23.jpg', 'f24.jpg', 'f25.jpg', 'f26.jpg', 'f27.jpg', 'f28.jpg', 'f29.jpg', 'f30.jpg', 'f31.jpg', 'f32.jpg', 'f33.jpg', 'f34.jpg', 'f35.jpg', 'f36.jpg', 'f37.jpg', 'f38.jpg', 'f39.jpg', 'f40.jpg', 'f41.jpg', 'f42.jpg', 'f43.jpg', 'f44.jpg', 'f45.jpg', 'f46.jpg', 'f47.jpg', 'f48.jpg', 'f49.jpg', 'f50.jpg', 'f51.jpg', 'f52.jpg', 'f53.jpg', 'f54.jpg', 'f55.jpg', 'f56.jpg', 'f57.jpg', 'f58.jpg', 'f59.jpg', 'f60.jpg', 'f61.jpg', 'f62.jpg', 'f63.jpg', 'f64.jpg', 'f65.jpg', 'f66.jpg', 'f67.jpg', 'f68.jpg', 'f69.jpg', 'f70.jpg', 'f71.jpg', 'f72.jpg', 'f73.jpg', 'f74.jpg', 'f75.jpg', 'f76.jpg', 'f77.jpg', 'f78.jpg', 'f79.jpg', 'f80.jpg', 'f81.jpg', 'f82.jpg', 'f83.jpg', 'f84.jpg', 'f85.jpg', 'f86.jpg', 'f87.jpg', 'f88.jpg', 'f89.jpg', 'f90.jpg', 'f91.jpg', 'f92.jpg', 'f93.jpg', 'f94.jpg', 'f95.jpg', 'f96.jpg', 'f97.jpg', 'f98.jpg', 'f99.jpg', 'f100.jpg', 'f101.jpg', 'f102.jpg', 'f103.jpg', 'f104.jpg', 'f105.jpg', 'f106.jpg', 'f107.jpg', 'f108.jpg', 'f109.jpg', 'f110.jpg', 'f111.jpg', 'f112.jpg', 'f113.jpg', 'f114.jpg', 'f115.jpg', 'f116.jpg', 'f117.jpg', 'f118.jpg', 'f119.jpg', 'f120.jpg' )
train <- list()
for (i in 1:360) {train[[i]] <- readImage(pic1[i])}

pic2 <- c('b121.jpg', 'c121.jpg', 'f121.jpg')
test <- list()
for (i in 1:3) {test[[i]] <- readImage(pic2[i])}

# Explore
print(train[[12]])
summary(train[[12]])
display(train[[12]])
plot(train[[12]])

par(mfrow = c(18,20))
for (i in 1:360) plot(train[[i]])
par(mfrow = c(1,1))

# Resize & combine
str(train)
for (i in 1:360) {train[[i]] <- resize(train[[i]], 42, 42)}
for (i in 1:3) {test[[i]] <- resize(test[[i]], 42, 42)}

train <- combine(train)
x <- tile(train, 20)
display(x, title='Pictures')

test <- combine(test)
y <- tile(test, 3)
display(y, title = 'Pics')

# Reorder dimension
train <- aperm(train, c(4, 1, 2, 3))
test <- aperm(test, c(4, 1, 2, 3))
str(train)

# Response
trainy <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
testy <- c(0, 1, 2)

# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# Model
model <- keras_model_sequential()

model %>%
         layer_conv_2d(filters = 20, 
                       kernel_size = c(5,5),
                       activation = 'relu',
                       input_shape = c(42,42,3)) %>%
         layer_max_pooling_2d(pool_size = c(2,2)) %>%
         layer_dropout(rate = 0.25) %>%
         layer_conv_2d(filters = 50,
                       kernel_size = c(15,15),
                       activation = 'relu') %>%
         layer_max_pooling_2d(pool_size = c(2,2)) %>%
         layer_dropout(rate = 0.25) %>%
         layer_flatten() %>%
         layer_dense(units = 500, activation = 'relu') %>%
         layer_dropout(rate=0.25) %>%
         layer_dense(units = 3, activation = 'softmax') %>%
         
         compile(loss = 'categorical_crossentropy',
                 optimizer = optimizer_sgd(learning_rate = 0.01,
                                           weight_decay = 1e-6,
                                           momentum = 0.9,
                                           nesterov = T),
                 metrics = c('accuracy'))
summary(model)

# Fit model
history <- model %>%
         fit(train,
             trainLabels,
             epochs = 100,
             batch = 20,
             validation_split = 0.2,
             validation_data = list(test, testLabels))
plot(history)

#########################################################
# Evaluation & Prediction - train data
model %>% evaluate(train, trainLabels)
model %>% predict(train) %>% k_argmax()

# Process the output on Ms. Word and paste the result
result <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1, 0, 1, 0, 2, 2, 0, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 2, 2, 0, 2, 2, 2, 0, 0, 1, 0)
trainy <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

library(caret)

predicted_value <- factor(result)
expected_value <- factor (trainy)

cm <- confusionMatrix(data=predicted_value, reference = expected_value)
cm
