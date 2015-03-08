dat <- read.csv("train.csv", header = T)
labels <- dat[,1]
test_idx <- c()
for(i in 1:10){
  tmp1 <- which(labels  == i-1)
  tmp2 <- sample(tmp1, 1000, replace = F)
  test_idx <- c(test_idx, tmp2)
}
test <- dat[test_idx, ]
train <- dat[-test_idx,]
write.table(train, file = "prac_train.csv", quote= F, col.names = T, row.names=F, sep = ",")
write.table(test, file = "prac_test.csv", quote = F, col.names = T, row.names = F, sep = ",")

# Just for visualization
test <- read.delim("prac_test.csv", sep = ",")
id <- rep(NA,10)
for(i in 0:9){
  id[i+1] <- which(test$label == i)[1]
}
par(mfrow = c(2,5))
for(i in 0:9){
  image(t(apply(matrix(as.vector(as.matrix(test[id[i+1], -1])), ncol = 28, nrow = 28,byrow = T), 2,rev)), col=grey(seq(1,0,length.out=256)))
}

library(h2o)
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = T, nthreads = -1)
trData<- h2o.importFile(localH2O, path = "prac_train.csv")
tsData<- h2o.importFile(localH2O, path = "prac_test.csv")

# in order to set a benchmark, we run a random forest classifier 
# as MNIST Kaggle competition recommends
prac_train <- read.csv("prac_train.csv")
prac_test <- read.csv("prac_test.csv")
library(randomForest)
prac_train$label <- as.factor(prac_train$label)
prac.rf <- randomForest(label ~ ., prac_train)
result_rc = table(prac_test$label, predict(prac.rf, newdata = prac_test[,-1]))
sum(diag(result_rc))

## overcome the benchmark with h2o.deeplearning
res.dl <- h2o.deeplearning(x = 2:785, y = 1, data = trData, activation = "Tanh", hidden=rep(160,5), epochs = 20)
pred.dl <- h2o.predict(object =  res.dl, newdata = tsData[,-1])
perd.dl.df <- as.data.frame(pred.dl)
sum(diag(table(prac_test$label, pred.dl.df[,1])))

## Hinton 2012
## activation: Tanh
## c(500,500,1000)
## epochs:20
## rate:0,01
## rate_annealing: 0.001


## h2o distributed deep learning by Arno Candel 071614
## Activation: RectifierWithDropout
## Hidden: c(1024, 1024, 2048)
## epochs: 200
## rate : 0.01
## rate_annealing: 1.0e-6
## rate_decay: 1.0
## momentum_start: 0.5
## momentum_ramp: 32000*12
## momentum_stable: .99
## input_dropout_ratio: .2
## l1:1.0e-5
## l2:0.0
## max_w2: 15
## initial_weight_distribution: Normal
## initial_weight_scale: .01
## nesterov_accelerated_gradient: True
## loss: CrossEntropy
## fast_mode: True
## diagnostics: True
## Ignore_const_cols: True
## force_load_balance: True
res.dl <- h2o.deeplearning(x = 2:785, y = 1, data = trData, activation = "RectifierWithDropout", hidden = c(1024, 1024,2048),epochs = 200, adaptive_rate = F, rate = .01, rate_annealing = 1.0e-6, rate_decay = 1.0, momentum_start = .5, momentum_ramp = 32000*12, momentum_stable = .99, input_dropout_ratio = .2, l1 = 1.0e-5, l2 = 0.0, max_w2 = 15.0, initial_weight_distribution = "Normal", initial_weight_scale = .01, nesterov_accelerated_gradient = T, loss = "CrossEntropy", fast_mode = T, diagnostics = T, force_load_balance = T)
pred.dl <- h2o.predict(object=res.dl, newdata=tsData[,-1])
pred.dl.df <- as.data.frame(pred.dl)
result = table(prac_test$label, pred.dl.df[,1])
sum(diag(result))

## training the dl model with all training data
## my pc can not produce a result.
ktrData <- h2o.importFile(localH2O, path = "train.csv")
ktsData <- h2o.importFile(localH2O, path = "test.csv")
res.dl <- h2o.deeplearning(x = 2:785, y = 1, data = trData, activation = "RectifierWithDropout", hidden = c(1024, 1024,2048),epochs = 200, adaptive_rate = F, rate = .01, rate_annealing = 1.0e-6, rate_decay = 1.0, momentum_start = .5, momentum_ramp = 32000*12, momentum_stable = .99, input_dropout_ratio = .2, l1 = 1.0e-5, l2 = 0.0, max_w2 = 15.0, initial_weight_distribution = "Normal", initial_weight_scale = .01, nesterov_accelerated_gradient = T, loss = "CrossEntropy", fast_mode = T, diagnostics = T, force_load_balance = T)
pred.dl <- h2o.predict(object = res.dl, newdata = ktsData)
pred.dl.df <- as.data.frame(pred.dl)

## write.table(pred.dl.df[,1], file = 'output.csv', quote = F, col.names = c("Label"), row.names = T, sep = ',')
ImageId <- 1:length(pred.dl.df[,1])
Label <- pred.dl.df[,1]
write.table(cbind(ImageId, Label) , file = "dl.csv", quote = F, col.names = T, row.names = F, sep = ',')
