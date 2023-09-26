library(xgboost)
library(readr)
library(stringr)
library(caret)
library(data.table)
library(xgboost)
library(caret)
library(stringr)
library(quanteda)
library(lubridate)
library(stringr)
library(Hmisc)
library(Matrix)




setwd("~/Desktop/457/project 2/w21projii")
trainDat = read.csv('W21P2_train.csv', header = TRUE)
testDat = read.csv('W21P2_test.csv', header = TRUE)
FtCols = 8:ncol(trainDat) #Feature columns for train
#Remove thos features with < 200 appearances
cntFts = colSums(trainDat[,FtCols]) + colSums(testDat[,FtCols])
selCols = (cntFts >= 200)
trainDat = trainDat[,c(1:6, FtCols[selCols])]
intClass = rep(1,nrow(trainDat))#1 for low, 2 for medium, 3 for high
intClass[trainDat$interest_level == 'medium'] = 2
intClass[trainDat$interest_level == 'low'] = 3
trainDat$interest_level = intClass
testDat = testDat[,c(2:6,FtCols[selCols])]
trainDat$Laundry.in.Unit = trainDat$Laundry.In.Unit + trainDat$Laundry.in.Unit
Laundry.In.Unit <- which(colnames(trainDat) == "Laundry.In.Unit")
testDat$Laundry.in.Unit = testDat$Laundry.In.Unit + testDat$Laundry.in.Unit
Laundry.In.Unit1 <- which(colnames(testDat) == "Laundry.In.Unit")
trainDat$Laundry.in.Building = trainDat$Laundry.In.Building + trainDat$Laundry.in.Building
Laundry.In.Building <- which(colnames(trainDat) == "Laundry.In.Building")
testDat$Laundry.in.Building= testDat$Laundry.In.Building + testDat$Laundry.in.Building
Laundry.In.Building1 <- which(colnames(testDat) == "Laundry.In.Building")
trainDat<-trainDat[,-c(Laundry.In.Unit,Laundry.In.Building)]
testDat<-testDat[,-c(Laundry.In.Unit1,Laundry.In.Building1)]
trainDat$Laundry.in.Building<-ifelse(trainDat$Laundry.in.Building == 0, 0, 1)
trainDat$Laundry.in.Unit<-ifelse(trainDat$Laundry.in.Unit == 0, 0, 1)
testDat$Laundry.in.Building<-ifelse(testDat$Laundry.in.Building == 0, 0, 1)
testDat$Laundry.in.Unit<-ifelse(testDat$Laundry.in.Unit == 0, 0, 1)
trainDat<-trainDat[-6174,]
trainDat<-trainDat[-c(4685,2541,7706),]
boxplot(price~interest_level,data=trainDat)
boxplot(latitude~interest_level,data=trainDat)
boxplot(longitude~interest_level,data=trainDat)
view(trainDat)
write.csv(trainDat, file = 'new_train1.csv', row.names = FALSE)
select_step = stepclass(interest_level~., data = trainDat, maxvar = 60,     method = "lda", criterion = "AC", direction = 'both')
select_step
trainDat<-trainDat[,-c(22,24,26,28,33,35)]
testDat<-testDat[,-c(21,23,25,27,32,34)]
write.csv(trainDat, file = 'new_train.csv', row.names = FALSE)
write.csv(testDat, file = 'new_test.csv', row.names = FALSE)
######################################Boosting###############################################################################
sample = sample(nrow(trainDat), round(nrow(trainDat)*0.75))
train = trainDat[sample, ]
test = trainDat[-sample, ]

train$interest_level=as.factor(train$interest_level)
ctrl <- trainControl(method = "cv",
                     number = 10,allowParallel = TRUE)

gbm.Grid = expand.grid(interaction.depth = 4, 
                       n.trees = 600, 
                       shrinkage = 0.05,
                       n.minobsinnode = 10) 
Model<-train(interest_level~ .,
             train,
             method='gbm',
             trControl=ctrl,
             tuneGrid=gbm.Grid)
Model$bestTune
gbm.pred = predict(Model, test)
gbm.pred<-make.names(gbm.pred)
A <- table(gbm.pred, test$interest_level)
sum(diag(A)) / sum(A)
#############################################################
trainDat$interest_level=as.factor(trainDat$interest_level)
ctrl <- trainControl(method = "cv",
                     number = 10,allowParallel = TRUE)


gbm.Grid = expand.grid(interaction.depth = c(2,3,4,5,6), 
                       n.trees = (1:5)*200, 
                       shrinkage = c(0.1, 0.05, 0.01),
                       n.minobsinnode = 10) 

Model<-train(interest_level~ .,
             trainDat,
             method='gbm',
             trControl=ctrl,
             tuneGrid=gbm.Grid)
Model$bestTune
summary(Model)
gbm.pred = predict(Model, testDat,type="prob")


out.pred = as.data.frame(gbm.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'boost000.csv', row.names = FALSE)


######################################################RF###################################################################
trainDat$interest_level=make.names(trainDat$interest_level)
sample = sample(nrow(trainDat), round(nrow(trainDat)*0.75))
train = trainDat[sample, ]
test = trainDat[-sample, ]

ctrl <- trainControl(method = "cv",
                     number = 10,classProbs = TRUE)

rf.Grid =  expand.grid(mtry = 10,
                       splitrule ='gini', 
                       min.node.size = 1) 


rf.cv.model <- train(interest_level ~ ., data = train,
                     method = "ranger",
                     trControl = ctrl,
                     tuneGrid = rf.Grid)

rf.cv.pre = predict(rf.cv.model, test)

A <- table(rf.cv.pre, test$interest_level)
sum(diag(A)) / sum(A)

ctrl <- trainControl(method = "cv",
                     number = 10,classProbs = TRUE)

rf.Grid =  expand.grid(mtry = 2*(1:10),
                       splitrule ='gini', 
                       min.node.size = 1) 

trainDat$interest_level<-make.names(trainDat$interest_level)
rf.cv.model <- train(interest_level ~ ., data = trainDat,
                     method = "ranger",
                     trControl = ctrl,
                     tuneGrid = rf.Grid)
plot(rf.cv.model)
#10
gbm.pred = predict(rf.cv.model, testDat,type="prob")

out.pred = as.data.frame(gbm.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'rf000.csv', row.names = FALSE)
##############################################################SVM#############################################################

ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = TRUE,
                     classProbs = TRUE)
rfSVM.Grid = expand.grid(C = 60, 
                         sigma = 0.004, 
                         Weight = 1)
registerDoParallel(cores=2) 
rfSVM.cv.model <- train( interest_level~ ., 
                         data = train,
                         method = "svmRadialWeights",
                         trControl = ctrl,
                         tuneGrid = rfSVM.Grid)

pred = predict(rfSVM.cv.model, test)

A <- table(pred, test$interest_level)
sum(diag(A)) / sum(A)



ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = TRUE,
                     classProbs = TRUE)
rfSVM.Grid = expand.grid(C = (2:6)*10, 
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
write.csv(out.pred, file = 'svm000.csv', row.names = FALSE)

select_step = stepclass(interest_level~., data = trainDat, maxvar = 60, 
                        method = "lda", criterion = "AC", direction = 'both')

###########################################################################With new variables#####################################
trainDat = read.csv('W21P2_train.csv', header = TRUE)
testDat = read.csv('W21P2_test.csv', header = TRUE)
FtCols = 8:ncol(trainDat) #Feature columns for train
#Remove thos features with < 200 appearances
cntFts = colSums(trainDat[,FtCols]) + colSums(testDat[,FtCols])
selCols = (cntFts >= 200)
trainDat = trainDat[,c(1:7, FtCols[selCols])]
intClass = rep(1,nrow(trainDat))#1 for low, 2 for medium, 3 for high
intClass[trainDat$interest_level == 'medium'] = 2
intClass[trainDat$interest_level == 'low'] = 3
trainDat$interest_level = intClass
testDat = testDat[,c(2:7,FtCols[selCols])]
trainDat$Laundry.in.Unit = trainDat$Laundry.In.Unit + trainDat$Laundry.in.Unit
Laundry.In.Unit <- which(colnames(trainDat) == "Laundry.In.Unit")
testDat$Laundry.in.Unit = testDat$Laundry.In.Unit + testDat$Laundry.in.Unit
Laundry.In.Unit1 <- which(colnames(testDat) == "Laundry.In.Unit")
trainDat$Laundry.in.Building = trainDat$Laundry.In.Building + trainDat$Laundry.in.Building
Laundry.In.Building <- which(colnames(trainDat) == "Laundry.In.Building")
testDat$Laundry.in.Building= testDat$Laundry.In.Building + testDat$Laundry.in.Building
Laundry.In.Building1 <- which(colnames(testDat) == "Laundry.In.Building")
trainDat<-trainDat[,-c(Laundry.In.Unit,Laundry.In.Building)]
testDat<-testDat[,-c(Laundry.In.Unit1,Laundry.In.Building1)]
trainDat$Laundry.in.Building<-ifelse(trainDat$Laundry.in.Building == 0, 0, 1)
trainDat$Laundry.in.Unit<-ifelse(trainDat$Laundry.in.Unit == 0, 0, 1)
testDat$Laundry.in.Building<-ifelse(testDat$Laundry.in.Building == 0, 0, 1)
testDat$Laundry.in.Unit<-ifelse(testDat$Laundry.in.Unit == 0, 0, 1)
trainDat<-trainDat[-6174,]
trainDat<-trainDat[-c(4685,2541,7706),]
boxplot(price~interest_level,data=trainDat)
boxplot(latitude~interest_level,data=trainDat)
boxplot(longitude~interest_level,data=trainDat)
view(trainDat)
write.csv(trainDat, file = 'new_train1.csv', row.names = FALSE)
select_step = stepclass(interest_level~., data = trainDat, maxvar = 60,     method = "lda", criterion = "AC", direction = 'both')
select_step
trainDat<-trainDat[,-c(23,25,27,29,34,36)]
testDat<-testDat[,-c(22,24,26,28,33,35)]
trainDat$street_address=as.integer(as.factor(trainDat$street_address))
testDat$street_address=as.integer(as.factor(testDat$street_address))

pricePerBed = ifelse(!is.finite(trainDat$price/trainDat$bedrooms),-1, trainDat$price/trainDat$bedrooms)
pricePerBath = ifelse(!is.finite(trainDat$price/trainDat$bathrooms),-1, trainDat$price/trainDat$bathrooms)
pricePerRoom = ifelse(!is.finite(trainDat$price/(trainDat$bedrooms+trainDat$bathrooms)),-1, trainDat$price/(trainDat$bedrooms+trainDat$bathrooms))
bedPerBath = ifelse(!is.finite(trainDat$bedrooms/trainDat$bathrooms), -1, trainDat$price/trainDat$bathrooms)
trainDat <- cbind(trainDat,pricePerBed,pricePerBath,pricePerRoom,bedPerBath)

pricePerBed = ifelse(!is.finite(testDat$price/testDat$bedrooms),-1, testDat$price/testDat$bedrooms)
pricePerBath = ifelse(!is.finite(testDat$price/testDat$bathrooms),-1, testDat$price/testDat$bathrooms)
pricePerRoom = ifelse(!is.finite(testDat$price/(testDat$bedrooms+testDat$bathrooms)),-1, testDat$price/(testDat$bedrooms+testDat$bathrooms))
bedPerBath = ifelse(!is.finite(testDat$bedrooms/testDat$bathrooms), -1, testDat$price/testDat$bathrooms)
testDat <- cbind(testDat,pricePerBed,pricePerBath,pricePerRoom,bedPerBath)
