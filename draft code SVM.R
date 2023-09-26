library(doParallel) 
library(rpart)
library(rpart.plot)
setwd("~/Desktop/457/project 2/w21projii")
set.seed(17)
trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)
names(trainDat)
trainDat<-trainDat[,-c(24,26,28,32,33,35)]
trainDat<-trainDat[-4685,]
names(testDat)
testDat<-testDat[,-c(23,25,27,31,32,34)]

trainDat$interest_level=as.factor(trainDat$interest_level)
trainDat$interest_level = make.names(trainDat$interest_level)

ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = TRUE,
                     classProbs = TRUE)
rfSVM.Grid = expand.grid(C = (1:6)*10, 
                         sigma = (1:8)*0.001, 
                         Weight = 1)
registerDoParallel(cores=2) 
rfSVM.cv.model <- train( interest_level~ ., 
                         data = trainDat,
                        method = "svmRadialWeights",
                        trControl = ctrl,
                        tuneGrid = rfSVM.Grid)

pred = predict(rfSVM.cv.model, testDat,type="prob")

out.pred = as.data.frame(pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'svm4.csv', row.names = FALSE)
select_step = stepclass(interest_level~., data = trainDat, maxvar = 60, 
                        method = "lda", criterion = "AC", direction = 'both')
trMod = rpart(interest_level~.^2, data = trainDat, method = 'class', control = list(cp=0))
printcp(trMod)
#########################################################################################
library(glmnet)
setwd("~/Desktop/457/project 2/w21projii")
setwd("~/Desktop/457/project 2/w21projii")
set.seed(17)
trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)
view(trainDat)
trainDat<-trainDat[,-c(24,26,28,32,33,35)]
names(testDat)
testDat<-testDat[,-c(23,25,27,31,32,34)]

trainDat$interest_level=as.factor(trainDat$interest_level)
trainDat$interest_level = make.names(trainDat$interest_level)


cvfit = cv.glmnet(as.matrix(trainDat[,-1]), as.matrix(trainDat[,1]), 
                  nfolds = 10, family="multinomial",
                  trace.it = 1, parallel = TRUE)

glmnet.pred = predict(cvfit, newx =  as.matrix(testDat), 
                      s = "lambda.min", type = "response")
out.pred = as.data.frame(glmnet.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'log1.csv', row.names = FALSE)


heartstepA = step(trainDat, scope=list(upper=~., lower=~1))
