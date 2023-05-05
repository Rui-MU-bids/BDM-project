library(caTools)
library(ggplot2)
library(caret)
library(glmnet)
library(MASS)
library(class)
library("pROC")
library(e1071)
library(randomForest)
library(wesanderson)


meps <- read.csv('~/Desktop/BDM final project/MEPS14.csv')
summary(meps$cost)
data <- meps[,c(2,9:19)]
dim(data)
for (i in 2:12){
  cat(paste0(colnames(data[i]),'\n'))
  print(table(data[,i]))
  cat('\n')
}

ggplot(data, aes(cost)) + geom_density(fill = 'lightblue') + ggtitle("Density Plot of Cost")

topcost <- quantile(data$cost, probs=0.95)
topcost
data$cost <- ifelse(data$cost>=topcost,1,0)
sum(data$cost==1)
data[1:10,]

set.seed(123)
data$cost <- as.factor(data$cost)
sample <- sample.split(data$cost, SplitRatio = 2/3)
train <- subset(data, sample == T)
test <- subset(data, sample == F)
ggplot(train, aes(cost)) + geom_bar(fill=c('lightblue','lightpink')) + ggtitle("Density Plot of Cost")

#balance train by oversampling
library(ROSE)
train <- ovun.sample(cost~. , data=train, method = 'over')
train <- train$data
ggplot(train, aes(cost)) + geom_bar(fill=c('lightblue','lightpink')) + ggtitle("Density Plot of Cost after Balancing")


for (i in 2:12){
  cat(paste0(colnames(train[i]),'\n'))
  print(table(train[,i]))
  cat('\n')
}



set.seed(123)
#Logistic
Logi <- glm(cost~. , data=train, family = 'binomial')
summary(Logi)
Logi.pred <- predict(Logi, test[,-1],type = 'response')
Logi.pred <- ifelse(Logi.pred > 0.5, 1, 0)
table(Logi.pred, test$cost)
mean(Logi.pred==test[,1])
roc(test$cost, as.numeric(Logi.pred))$auc




#LDA
lda.fit <- lda(cost~., train)
lda.pred <- predict(lda.fit, test[,-1])$class
table(lda.pred, test$cost)
fourfoldplot(table(lda.pred, test$cost), color=c('lightpink','mistyrose'),conf.level=0, margin = 1,main = 'LDA')
mean(lda.pred==test[,1])
confusionMatrix(lda.pred, test$cost)
?confusionMatrix
roc(test$cost, as.numeric(lda.pred))$auc

#QDA
qda.fit <- qda(cost~. ,train)
qda.pred <- predict(qda.fit, test[,-1])$class
table(qda.pred, test$cost)
fourfoldplot(table(qda.pred, test$cost), color=c('lightpink','mistyrose'),conf.level=0, margin = 1,main = 'QDA')
mean(qda.pred==test[,1])
confusionMatrix(qda.pred, test$cost)
roc(test$cost, as.numeric(qda.pred))$auc
roc(test$cost, as.numeric(qda.pred),plot=T)

#SVM
svm.fit <- svm(cost~. , train, kernel='linear', cost=10, scale = F)
svm.pred <- predict(svm.fit, test)
table(svm.pred, test$cost)
fourfoldplot(table(svm.pred, test$cost), color=c('lightpink','mistyrose'),conf.level=0, margin = 1,main = 'SVM-kernel')
mean(svm.pred==test$cost)
confusionMatrix(svm.pred, test$cost)
roc(test$cost, as.numeric(svm.pred))$auc

svm.fit <- svm(cost~. , train, kernel='radial', cost=10, scale = F)
svm.pred <- predict(svm.fit, test)
table(svm.pred, test$cost)
fourfoldplot(table(svm.pred, test$cost), color=c('lightpink','mistyrose'),conf.level=0, margin = 1,main = 'SVM-radial')
mean(svm.pred==test$cost)
confusionMatrix(svm.pred, test$cost)
roc(test$cost, as.numeric(svm.pred))$auc

#randomforest
rm <- randomForest(cost~ . , data=train, ntree=1000, mtry = 1, importance=T)
rm.pred <- predict(rm, test)
rm.pred2 <- predict(rm,train)
mean(rm.pred2==train$cost)
table(rm.pred, test$cost)
fourfoldplot(table(rm.pred, test$cost), color=c('lightpink','mistyrose'),conf.level=0, margin = 1,main = 'RandomForest')
mean(rm.pred==test$cost)
confusionMatrix(rm.pred, test$cost)
roc(test$cost, as.numeric(rm.pred))$auc
varImpPlot(rm)

#tree
library(tree)
tree <- tree(cost~ . , data=train)
plot(tree)
text(tree)
cv <- cv.tree(tree, FUN=prune.misclass)
plot(cv$size,cv$dev,xlab='tree size', ylab = 'Classification Error', type = 'b')
prune <- prune.misclass(tree, best=7)
pred.prune <- predict(prune, newdata=test, type = 'class')
mean(pred.prune==test$cost)
roc(test$cost, as.numeric(pred.prune))$auc

#knn
train.x <- train[,-1]
train.y <- train[,1]
test.x <- test[,-1]
knn <- knn(train.x, test.x, train.y, k=1)
table(knn, test[,1])
fourfoldplot(table(knn, test[,1]), color=c('lightpink','mistyrose'),conf.level=0, margin = 1,main = 'KNN')
mean(knn==test$cost)
roc(test$cost, as.numeric(knn))$auc

summary(train$cost)

ggplot(train, aes(x = cost,y = as.factor(DIABETES), col=cost)) + geom_jitter()




#add INSCOV
data2 <- meps[,c(2,9:19,33)]
data2$cost <- ifelse(data2$cost>=topcost,1,0)
set.seed(123)
data2$cost <- as.factor(data2$cost)
sample <- sample.split(data2$cost, SplitRatio = 2/3)
train2 <- subset(data2, sample == T)
test2 <- subset(data2, sample == F)
train2 <- ovun.sample(cost~. , data=train2, method = 'over')
train2 <- train2$data
ggplot(train2, aes(cost)) + geom_bar(fill=c('lightblue','lightpink')) + ggtitle("Density Plot of Cost after Balancing")
Logi2 <- glm(cost~. , data=train2, family = 'binomial')
summary(Logi)
lda.fit2 <- lda(cost~., train2)
lda.pred2 <- predict(lda.fit2, test2[,-1])$class
confusionMatrix(lda.pred2, test2$cost)
roc(test2$cost, as.numeric(lda.pred2))$auc



