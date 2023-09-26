library(ranger)
setwd("~/Desktop/457/project 2/w21projii")
train = read.csv('W21P2_train.csv', header = TRUE)
test = read.csv('W21P2_test.csv', header = TRUE)

trainDat = read.csv('new_train.csv', header = TRUE)
testDat = read.csv('new_test.csv', header = TRUE)

trainDat<-trainDat[,-c(24,26,28,32,33,35)]
testDat<-testDat[,-c(23,25,27,31,32,34)]
Data<-trainDat[,1:6]
summary(Data)
cor(Data)
p1<-boxplot(price~interest_level,data=train)
p2<-boxplot(latitude~interest_level,data=train)
p3<-boxplot(longitude~interest_level,data=train)
par(mfrow=c(2,2))
boxplot(price~interest_level,data=train)
boxplot(latitude~interest_level,data=train)
boxplot(longitude~interest_level,data=train)


names(trainDat)
view(trainDat)
trainDat<-trainDat[-4685,]

trainDat$interest_level = make.names(trainDat$interest_level)

ctrl <- trainControl(method = "cv",
                     number = 10,classProbs = TRUE)

rf.Grid =  expand.grid(mtry = 2*(1:10),
                                splitrule ='gini', 
                                min.node.size = 1) 


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
write.csv(out.pred, file = 'rf8.csv', row.names = FALSE)
