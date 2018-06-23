rm(list = ls(all=TRUE))
setwd("C:\\Users\\rvadamala\\Documents\\insofe\\CUTE_CS7305C")
dir()

## Reading the data
inputData <-  read.csv("train.csv",header = T, na.strings = "NA")
testData <- read.csv("test.csv",header=T,na.strings = "NA")

str(inputData)
summary(inputData)

## Understanding of data

#1) ID is unique column, make the row names as ID
rownames(inputData) <- inputData$ID 
inputData$ID <-  NULL

#2) target is categorical , lets change that to factor datatype
inputData$target <- as.factor(as.character(inputData$target))

#3) Mean is very much far from Median for most of the attributes, that implies existence of outliers
### Let's look into it .. how many outliers are there in each column 

##----------------------------------------------------------##
##      Function to find percentage of Outliers in data     ##
##----------------------------------------------------------##


findNumOfOutliers <- function(x) {
  noOfColumns <- ncol(x)
  count <- c()
  for(col in 1:noOfColumns) {
    outliers <- boxplot.stats(x[,col])$out
    count <- c(count,length(outliers))
    
  }
  count <- ((count*100)/nrow(x))
  names(count) <- colnames(x)
  count <- sort(count,decreasing = T)
  
  return(count)
}

outlierCount <- findNumOfOutliers(inputData[,-ncol(inputData)])
print(outlierCount)

## Observations : It is observed that there are lot of outliers in this data.. but when we look into the business problem
## it is to found the bankruptcy of the firms..may be these are the values which will make sense to show a bankrupted company
## so, lets keep it for now.. and may be before modelling.. we may need to impose a method to do feature selection and 
## remove the insignificant variables.

## Any Missing values ??

sum(is.na(inputData))

##----------------------------------------------------------##
##    Function to find columns with more than 20% NAs       ##
##----------------------------------------------------------##

findRemovableNaColumns <- function(x){
  colSums <- (colSums(is.na(x)) / nrow(x) ) * 100
  names(colSums) <- colnames(x)
  colSums <- sort(colSums,decreasing = T)
  colNames <- names(colSums[colSums > 20])
  return (colNames)
}

naColsToBeRemoved <- findRemovableNaColumns(inputData[,-ncol(inputData)])
inputData <- inputData[,!(names(inputData) %in% naColsToBeRemoved)]
testData <- testData[,!(names(testData) %in% naColsToBeRemoved)]

## Imputing the remaining columns with centralImputation 
## Reason : centralImputation works with median, which is less affected by outliers

inputData <- DMwR::centralImputation(inputData)
testData <- DMwR::centralImputation(testData)
summary(inputData)
sum(is.na(inputData))


sum(is.na(testData))


## lets do train/validate split
## using createDataPartition method for stratified sampling

set.seed(1512)
trainRows <- caret::createDataPartition(inputData$target, p = 0.8, list = F)
trainData <- inputData[trainRows,]
validateData <- inputData[-trainRows,]

## check for class imbalance in train and validate splits
table(inputData$target)
table(trainData$target)
table(validateData$target)


## range of all attributes seems varying a lot.. as there are lot of features, may be its intuitive to try feature
## selection. Lets bring all features to same level and see ..lets standardize !

stdObj <- caret::preProcess(trainData[,-ncol(trainData)], method = c("center","scale"))
trainData <- predict(stdObj,trainData)
validateData <- predict(stdObj,validateData)
testData <- predict(stdObj,testData)

summary(trainData)
summary(validateData)

## Observation : After standardizing and imputation with median, the effect of outliers is reduced , it can be noticed
## with mean and median variations

## For feature selection, trying out PCA as we can remove the smaller variant dimensions, which may be noise

## prepare covariance matrix for pca input
pca_trainData <- princomp(trainData[,-ncol(trainData)])

print(pca_trainData$loadings)
head(pca_trainData$scores)

plot(pca_trainData)

trainPcaData <- data.frame(pca_trainData$scores,"target" = trainData$target)
par(mfrow=c(1,1))
plot(trainPcaData$Comp.1,trainPcaData$Comp.2,col=trainPcaData$target)
## variance seems exactly same , may be not a better option for feature selection

print(pca_trainData$loadings)

## Method 2 : corrplot

corrplot::corrplot(cor(inputData[,-ncol(inputData)]), method = "shade", type= "full" )

## observations.. lot of strong correlations found.. lets try VIF to drop any highly correlated variables

## lets build basic glm model 

glmBasicModel <- glm(target~.,trainData,family = "binomial")
summary(glmBasicModel)


## glm is showing all as significant, lets try vif and see

vifValues <- car::vif(glmBasicModel)
colsWithHighCorrelation <- names(vifValues[vifValues>5])


## from the corrplot, we can assume the pair of attributes that are showing high correlation

## Building VIF model with insignificant params removed

vifModel <- glm(target ~ ., trainData[,!(names(trainData) %in% colsWithHighCorrelation)], family="binomial")
summary(vifModel)

## prediction of validation accuracy w.r.t glmbasic and vif models

probTrainGlm <- predict(glmBasicModel,trainData)
predsTrainGlm <- ROCR::prediction(probTrainGlm,trainData$target)
perfTrainGlm <- ROCR::performance(predsTrainGlm, measure = "tpr",x.measure = "fpr")

ROCR::plot(perfTrainGlm, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
predsTrainOp <- ifelse(probTrainGlm > 0.2, "1","0")
caret::confusionMatrix(predsTrainOp,trainData$target,positive="1")


probValidateData <- predict(glmBasicModel,validateData,type="response")
predsValidateGlmOP <- ifelse(probValidateData >0.2,"1","0")

caret::confusionMatrix(predsValidateGlmOP, validateData$target, positive="1")
probTrainVif <- predict(vifModel,trainData)
predsTrainVif <- ROCR::prediction(probTrainVif,trainData$target)
perfTrainVif <- ROCR::performance(predsTrainVif,measure = "tpr",x.measure = "fpr")

ROCR::plot(perfTrainVif, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,0.1,0.01))
predTrainOpVif <- ifelse(probTrainVif>0.5,"1","0")
findTestPredictions(vifModel,0.5)
## Applying vif on train and validation

#findTestPredictions(glmBasicModel,0.2)

findTestPredictions <- function(x,y){
  probsTestData <- predict(x,testData,type="response")
  predsTestGlmOp <- ifelse(probsTestData > y, 1, 0)
  testOutputDf <- data.frame(testData$ID, predsTestGlmOp)
  colnames(testOutputDf) <- c("ID","prediction")
  write.csv(testOutputDf,file = "submission.csv" )
}

probTestUpload <- predict(vifModel,testData, type="response",norm.votes = F)
predsTestUpload <- ifelse(probTestUpload > 0.2,1,0)

output <- data.frame("ID"=testData$ID, "prediction"=probTestUpload)
write.csv(output, file="SampleSubmission.csv")

## glm doesnt seem to be performing well on test data.. one of the reasons ,might be
## class imbalance between the 1s and 0s. Treating bankruptcy as a rare case compared to 
## not bankrupt case.. and hence ..the distribution of 1s and 0s is ~ 20:1

prop.table(table(inputData$target))

## Lets look at some options to handle imbalanced data
## The metric to analyse model complexity is F1 score, so we need to look at precision and recall
## and our data is biased to 0s ( Not bankrupt case). lets use crossvalidation to train our model
## with rose sampling


#trainControlRf <- caret::trainControl(method = "repeatedcv",
#                                      number = 10,
#                                      repeats = 10,
#                                      sampling = "rose")
#set.seed(625)
#randomForestModelRose <- caret::train(target~., trainData, method="rf", trControl = trainControlRf)


## Handling imbalance of data using ROSE package
rose_train_data <- ROSE::ROSE(target~.,trainData, seed=777)$data
print(table(rose_train_data$target))

## synthetic data sampling done thru ROSE package
## lets run glm on rose data

glm_rose_model <- glm(target~., rose_train_data, family = "binomial")
summary(glm_rose_model)

rose_validate_preds <- predict(glm_rose_model, validateData, type="response")
validate_preds <- ifelse(rose_validate_preds > 0.5 , 1, 0)
caret::confusionMatrix(validate_preds,validateData$target)
install.packages("pROC")
roc_curver <- ROSE::roc.curve(validateData$target, validate_preds)
print(roc_curver)


## prediction on test data
rose_test_preds <- predict(glm_rose_model, testData, type="response")
test_preds_rose <- ifelse(rose_test_preds >0.4,1,0)

output <- data.frame("ID"=testData$ID, "prediction"=test_preds_rose)
write.csv(output, file="SampleRoseSubmission.csv")
table(test_preds_rose)

confMatrix <- caret::confusionMatrix(validate_preds,validateData$target,mode = 'prec_recall', positive = "1")
class(confMatrix)


## doing a grid search for best F1 score in validation data threshold
fscorelist <- data.frame()
for(i in seq(0.1,0.6,0.01))
{
  predsTest <- ifelse(rose_validate_preds > i , 1, 0)
  confMatrix <- caret::confusionMatrix(predsTest,validateData$target,positive = "1")
  fscorelist <- rbind(fscorelist,data.frame(i,confMatrix$byClass[[7]]))
}


names(fscorelist)<- c("threshold","fscore")
chosen_threshold<-fscorelist$threshold[fscorelist$fscore==max(fscorelist[,2])]

test_preds_rose <- ifelse(rose_test_preds >chosen_threshold,1,0)

output <- data.frame("ID"=testData$ID, "prediction"=test_preds_rose)
write.csv(output, file="SampleRoseSubmission.csv")
# max 13.5%

## data seems to be very variant. hence we need to choose a model which can handle high variance
## RandomForest


set.seed(777)
randomForestModel <- randomForest::randomForest(target~.,data=trainData, keep.forest=T,ntree=100)
print(randomForestModel)

randomForestModel$importance
round(randomForest::importance(randomForestModel),2)

rf_Imp_Attr <- data.frame(randomForestModel$importance)
rf_Imp_Attr <- data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) <- c('Attributes','Importance')
rf_Imp_Attr <- rf_Imp_Attr[order(rf_Imp_Attr$Importance,decreasing = T),]


randomForest::varImpPlot(randomForestModel)
## predicting test results with random forest model
predsTestRf <- predict(randomForestModel, testData[,setdiff(names(testData), "target")], type = "response", norm.votes = TRUE)


outputRf <- data.frame("ID"=testData$ID, "prediction"=predsTestRf)
write.csv(output, file="SampleRfSubmission.csv")

##13.1 %

## the top 30 variables are seems to be important (>25 range)
set.seed(777)
topImpAttr <- as.character(rf_Imp_Attr$Attributes[1:20])
rfModelImp <-  randomForest::randomForest(target~.,data=trainData[,c(topImpAttr,"target")], keep.forest=TRUE,ntree=100)
print(rfModelImp)

predsTestImpRf <- predict(rfModelImp, testData[,topImpAttr],type="response",norm.votes = TRUE)
outputImpRf <- data.frame("ID"=testData$ID, "prediction"=predsTestImpRf)
write.csv(outputImpRf, file="SampleImpRfSubmission.csv")

## 9.4 %
## Recall is 100% ( all positives are predicted fine, negatives are getting missed)

## lets tune our RF and see 




