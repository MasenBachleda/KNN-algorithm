setwd("~/Desktop/854datasets")
universal.df <- read.csv("UniversalBank.csv", header= TRUE)

library(fastDummies)
universal.df <- dummy_cols(universal.df, select_columns = 'Education')

universal.df <- universal.df[,-8]
head(universal.df)
train.index <- sample(rownames(universal.df), 0.6*dim(universal.df)[1])
valid.index <- setdiff(rownames(universal.df), train.index)

train.df <- universal.df[train.index,]
valid.df <- universal.df[valid.index,]

train.df <- train.df[, -c(1,5)]
valid.df <- valid.df[, -c(1,5)]
univ.df <- universal.df[, -c(1,5)]

str(univ.df)
new.df <- data.frame(Age=40, Experience=10, Income=84, Family=2, CCAvg=2, Mortgage=0, Securities.Account=0, CD.Account=0, Online=1, CreditCard=1, Education_1=0, Education_2=1, Education_3=0)

train.norm.df <- train.df
valid.norm.df <- valid.df
univ.norm.df <- univ.df

norm.values <- preProcess(train.df[, -7], method=c("center", "scale"))
train.norm.df[, -7] <- predict(norm.values, train.df[, -7])
valid.norm.df[, -7] <- predict(norm.values, valid.df[, -7])
univ.norm.df[, -7] <- predict(norm.values, univ.df[, -7])
valid.norm.df$Personal.Loan <- as.factor(valid.norm.df$Personal.Loan)
new.norm.df <- predict(norm.values, new.df)

library(FNN)
nn <- knn(train=train.norm.df[, -7], test=new.norm.df, cl=train.norm.df$Personal.Loan, k=1)
row.names(train.df)[attr(nn, "nn.index")]
nn
# customer will be classified as a non successful loan acceptance with k=1

accuracy.df <- data.frame(k=seq(1, 20, 1), accuracy=rep(0, 20))
for (i in 1:20){
  knn.pred <- knn(train.norm.df[, -7], valid.norm.df[, -7], cl=train.norm.df$Personal.Loan, k=i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.norm.df$Personal.Loan)$overall[1]
}  
accuracy.df
#according to the accuracy data frame, 3 is the best choice for k which has highest accuracy

knn.pred2 <- knn(train.norm.df[, -7], valid.norm.df[, -7], cl=train.norm.df$Personal.Loan, k=3)
confusionMatrix(knn.pred2, valid.norm.df$Personal.Loan)

knn.pred.new <- knn(train.norm.df[, -7], new.norm.df, cl=train.norm.df$Personal.Loan, k=3)
knn.pred.new
# this customer would be classified as non successful loan acceptance

train.index2 <- sample(rownames(universal.df), 0.5*dim(universal.df)[1])
valid.index2 <- sample(setdiff(rownames(universal.df), train.index2), 0.3*dim(universal.df)[1])
test.index <- setdiff(rownames(universal.df), union(train.index2, valid.index2))

train2 <- universal.df[train.index2,]
valid2 <- universal.df[valid.index2,]
test2 <- universal.df[test.index,]

train2 <- train2[, -c(1,5)]
valid2 <- valid2[, -c(1,5)]
test2 <- test2[, -c(1,5)]

train2.norm <- train2
valid2.norm <- valid2
test2.norm <- test2

norm2 <-  preProcess(train2[, -7], method=c("center", "scale"))
train2.norm[,-7] <- predict(norm2, train2[,-7])
valid2.norm[,-7] <- predict(norm2, valid2[,-7])
test2.norm[,-7] <- predict(norm2, test2[, -7])
valid2.norm$Personal.Loan <- as.factor(valid2.norm$Personal.Loan)
test2.norm$Personal.Loan <- as.factor(test2.norm$Personal.Loan)
new.norm2 <- predict(norm2, new.df)

knn.pred3 <- knn(train2.norm[,-7], valid2.norm[,-7], cl=train2.norm$Personal.Loan, k=3)
confusionMatrix(knn.pred3, valid2.norm$Personal.Loan)  

knn.pred4 <- knn(train2.norm[,-7], test2.norm[,-7], cl=train2.norm$Personal.Loan, k=3) 
confusionMatrix(knn.pred4, test2.norm$Personal.Loan)  
# confusion matrix with the validation set knn model is more accurate 
# the validation could be slightly more accurate because the sample size is 10% larger


