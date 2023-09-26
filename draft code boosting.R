library(Hmisc)
library(tidyverse)
library(caret)
library(ggplot2)
library(cowplot)
library(gbm)
library(klaR)
library(xgboost)
library(GGally) 
setwd("~/Desktop/457/project 2/w21projii")
set.seed(17)
train = read.csv('W21P2_train.csv', header = TRUE)
test = read.csv('W21P2_test.csv', header = TRUE)




view(trainDat)
str(trainDat)

summary(trainDat$interest_level)
p1<-ggplot(data=trainDat,aes(x=interest_level))+
  geom_bar(fill="lightblue")
p2<-trainDat%>%
  group_by(bedrooms)%>%
  summarize(mean_price=mean(price))%>%
  ggplot(aes(x=bedrooms,y=mean_price))+
  geom_col(fill="lightblue")

p3<-ggplot(trainDat,aes(x=interest_level,fill=factor(bedrooms)))+
  geom_bar(position="dodge")

p4<-ggplot(trainDat,aes(x=interest_level,fill=factor(bathrooms)))+
  geom_bar(position="dodge")
plot_grid(p1,p2,p3,p4,labels = c('interest', 'bedrooms vs mean price','interest & bedrooms','interest & bathrooms'))
?plot_grid()
################################################################################
ggplot(trainDat,mapping=aes(x=price))+
  geom_histogram()

##################################################################################


setwd("~/Desktop/457/project 2/w21projii")
set.seed(17)
trainDat = read.csv('W21P2_train.csv', header = TRUE)
testDat = read.csv('W21P2_test.csv', header = TRUE)


FtCols = 8:ncol(trainDat) #Feature columns for train

#Remove thos features with < 200 appearances
cntFts = colSums(trainDat[,FtCols]) + colSums(testDat[,FtCols])
names(trainDat)
selCols = (cntFts >= 200)
trainDat = trainDat[,c(1:6, FtCols[selCols])]
intClass = rep(1,nrow(trainDat))#1 for low, 2 for medium, 3 for high
intClass[trainDat$interest_level == 'medium'] = 2
intClass[trainDat$interest_level == 'low'] = 3
trainDat$interest_level = intClass

testDat = testDat[,c(2:6,FtCols[selCols])]
dim(trainDat)

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
names(testDat)
dim(trainDat)
trainDat<-trainDat[-6174,]
dim(testDat)
#8830   35
write.csv(trainDat, file = 'new_train.csv', row.names = FALSE)
write.csv(testDat, file = 'new_test.csv', row.names = FALSE)
##############################################################################
trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)
str(trainDat)
str(testDat)
library(doParallel) #install first
registerDoParallel(cores=2)




library(tictoc)
tic()
trainDat$interest_level = as.factor(trainDat$interest_level)

gbm.cv.model <- train(interest_level ~ ., data = trainDat,
                      method = "gbm",
                      trControl = ctrl,
                      tuneGrid = gbm.Grid,
                      verbose = FALSE)
?train
gbm.cv.model$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#37     400                 4      0.05             10

gbm.mod=gbm(interest_level~., data = trainDat, 
            distribution="multinomial", 
            n.trees=400, cv.folds = 5)
gbm.perf(gbm.mod, method = "cv")
gbm.pred = predict(gbm.mod, testDat, type = 'response') 


boost.cv.pred = predict(gbm.model, newdata =testDat )
out.pred = as.data.frame(gbm.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'boost1.csv', row.names = FALSE)
#1.00510
#############################################################################
setwd("~/Desktop/457/project 2/w21projii")
set.seed(17)
trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)

trainDat$interest_level = as.factor(trainDat$interest_level)


sample = sample(nrow(trainDat), round(nrow(trainDat)*0.75))
train = trainDat[sample, ]
test = trainDat[-sample, ]

library(tictoc)
tic()

train$interest_level = as.factor(train$interest_level)
library(doParallel) #install first
registerDoParallel(cores=2) 
ctrl <- trainControl(method = "cv",
                     number = 10,allowParallel = TRUE)

gbm.Grid = expand.grid(interaction.depth = c(2,3,4,5,6), 
                       n.trees = (1:5)*200, 
                       shrinkage = c(0.1, 0.05, 0.01),
                       n.minobsinnode = 10) 

train[,1]=make.names(train[,1])
test[,1]=make.names(test[,1])

Model<-train(interest_level~.,trainDat[,-7],
             method='gbm',
             trControl=ctrl,
             tuneGrid=gbm.Grid)
gbm.pred = predict(Model, test)
names(test)
names(train)
mean(gbm.pred!= test[,1])
#0.5167572
#0.5131341

gbm.pred = predict(Model, testDat,type="prob")

trainDat$interest_level = as.factor(trainDat$interest_level)
testDat$interest_level = as.factor(testDat$interest_level)
library(doParallel) #install first
registerDoParallel(cores=2) 
ctrl <- trainControl(method = "cv",
                     number = 10,allowParallel = TRUE)

gbm.Grid = expand.grid(interaction.depth = c(2,3,4,5,6), 
                       n.trees = (1:5)*200, 
                       shrinkage = c(0.1, 0.05, 0.01),
                       n.minobsinnode = 10) 

train[,1]=make.names(train[,1])
test[,1]=make.names(test[,1])

Model<-train(interest_level~.,train[,-7],
             method='gbm',
             trControl=ctrl,
             tuneGrid=gbm.Grid)
gbm.pred = predict(Model, test)


out.pred = as.data.frame(gbm.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'boost2.csv', row.names = FALSE)
#0.99


###########################################################################
setwd("~/Desktop/457/project 2/w21projii")
set.seed(17)
boxplot(price~interest_level,data=train)
trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)
trainDat<-trainDat[-6174,]
trainDat$interest_level = as.factor(trainDat$interest_level)

ctrl <- trainControl(method = "cv",
                     number = 10,allowParallel = TRUE)

gbm.Grid = expand.grid(interaction.depth = c(2,3,4,5,6), 
                       n.trees = (1:5)*200, 
                       shrinkage = c(0.1, 0.05, 0.01),
                       n.minobsinnode = 10) 

Model<-train(interest_level~ Balcony+bathrooms + bedrooms + price + Doorman + 
               Elevator + Laundry.in.Building + Dishwasher + Hardwood.Floors + No.Fee + 
               Reduced.Fee + Common.Outdoor.Space + Cats.Allowed + Dogs.Allowed + Fitness.Center + 
               Laundry.in.Unit + Pre.War  + Exclusive + High.Speed.Internet + 
               Dining.Room + Outdoor.Space + Furnished + 
               Loft + Wheelchair.Access,
             trainDat,
             method='gbm',
             trControl=ctrl,
             tuneGrid=gbm.Grid)

gbm.pred = predict(Model, testDat,type="prob")


out.pred = as.data.frame(gbm.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'boost3.csv', row.names = FALSE)

#######################################################################################
setwd("~/Desktop/457/project 2/w21projii")
trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)
names(trainDat)
trainDat<-trainDat[,-c(24,26,28,32,33,35)]
names(testDat)
testDat<-testDat[,-c(23,25,27,31,32,34)]
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
#   n.trees interaction.depth shrinkage n.minobsinnode
#38     600                 4      0.05             10
gbm.pred = predict(Model, testDat,type="prob")


out.pred = as.data.frame(gbm.pred)
colnames(out.pred) = c('high','medium','low')
out.pred = cbind(data.frame(ID = 1:nrow(testDat)), out.pred)
write.csv(out.pred, file = 'boost4.csv', row.names = FALSE)
