library(tidyverse)
library(dplyr)
library(ggplot2)
library(readxl)
library(dummies)    
library(philentropy)
library(skimr)     
library(cluster)    ## datasets and utilities for clustering, along with some algos
library(factoextra)    ## clustering visualization utilities
library(dendextend)     ## for working with dendrograms
library(devtools) 
# install.packages("gbm")
# install.packages("rpart.plot")
# install.packages("glmnet")
library(randomForest)
library(gbm)
library(rpart.plot)
library(rpart)
library(dplyr)
library(readr)
library(tidyverse) 
library(ggplot2) 
library(ggthemes)
library(glmnet)
library(MASS)
library(caret) # models
library(corrplot) # correlation plots
library(DALEX) # explain models
library(DescTools) # plots
library(doParallel) # parellel processing
library(dplyr) # syntax
library(inspectdf) # data overview
library(readr) # quick load
library(sjPlot) # contingency tables
library(tabplot) # data overview
library(tictoc) # measure time
library(MLmetrics)

#  boosting
bank = read_csv("bankdata_cleaned.csv")
set.seed(123)
bank$train <- sample(c(0, 1), nrow(bank), replace = TRUE, prob = c(.3, .7))
bank$deposit = NULL
bank$job = NULL
bank$marital = NULL
glimpse(bank)
bk_test <- bank %>% filter(train == 0)
bk_train <- bank %>% filter(train == 1) 
bk_train$train = NULL     
bk_test$train = NULL  

f1 = as.formula(Deposit ~age +
                  education+
                  default+
                  balance+
                  housing+
                  loan+
                  duration+
                  campaign+
                  pdays+
                  previous+
                  job_admin+
                  job_blue_collar+
                  job_entrepreneur+
                  job_housemaid+
                  job_management+
                  job_retired+
                  job_self_emply+
                  job_services+
                  job_student +
                  job_technician+
                  job_unemply+
                  m_divorced+
                  m_married+
                  m_single)

#Random Forest
Deposit_rf <- randomForest(f1, bk_train, ntree = 1000, do.trace=F) 
yhat_rf_train <- predict(Deposit_rf, bk_train)
mse_rf_train <- mean((yhat_rf_train - bk_train$Deposit) ^2) 
yhat_rf_test <- predict(Deposit_rf, bk_test)
mse_rf_test <- mean((yhat_rf_test - bk_test$Deposit) ^2) 
varImpPlot(Deposit_rf)
mean(bank$duration)

paste("Random Forest Train MSE",mse_rf_train)
paste("Random Forest Train MSE",mse_rf_test)

bank$DurationRange<-findInterval(bank$duration, c(0,100,200,300,400,500,600,800,1000,1500,2000,2500,3000))
bank1 <-bank %>% group_by(DurationRange) %>% summarise(deposit=mean(Deposit))
ggplot(bank1)+geom_col(aes(DurationRange,deposit))+scale_x_discrete(
  limits=c("0","100","200","300","400","500","600","800","1000","1500","2000","2500","3000"))


summary(bank1)

# boosting
fit.tree <- rpart(f1,
                  bk_train,
                  control = rpart.control(cp = 0.01))

fit_btree <- gbm(f1,
                 data = bk_train,
                 distribution = "gaussian",
                 n.trees = 1000,
                 interaction.depth = 2,
                 shrinkage = 0.01)
relative.influence(fit_btree)

# MSE for train
yhat_btree_train<- predict(fit_btree, bk_train, n.trees = 1000)
mse_btree_train <- mean((yhat_btree_train - bk_train$Deposit) ^ 2)
print(mse_btree_train)

# MSE for test
yhat_btree_test <- predict(fit_btree, bk_test, n.trees = 1000)
mse_btree_test <- mean((yhat_btree_test - bk_test$Deposit) ^ 2)
print(mse_btree_test)

##########################################  Binary Classification method 
df = read_csv("bank-original.csv")
df$pdays = NULL #Drop the variable `pdays` because it is misleading
df$duration = NULL 
glimpse(df)
# train model
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
# Define model setting 
# train and test 
set.seed(123)
trainIndex <- createDataPartition(df$deposit,
                                  p = 0.8, # training contains 80% of data
                                  list = FALSE)
dfTrain <- df[ trainIndex,]
dfTest  <- df[-trainIndex,]
set.seed(123)
# splitting
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary) # return more metrics than binary classification

# parameter grid for XGBoost
parameterGrid <-  expand.grid(eta = 0.1, # shrinkage (learning rate)
                              colsample_bytree = c(0.5,0.7), # subsample ration of columns
                              max_depth = c(3,6), # max tree depth. model complexity
                              nrounds = 10, # boosting iterations
                              gamma = 1, # minimum loss reduction
                              subsample = 0.8, # ratio of the training instances
                              min_child_weight = 2) # minimum sum of instance weight

# parameter grid for random forest
# mtry = the number of features to use to build each tree
rfGrid <- expand.grid(mtry = seq(from = 4, to = 20, by = 4))

# logistic regression
set.seed(123) 
model_glm <- train(deposit~.,
                   data = dfTrain,
                   method = "glm",
                   family = "binomial",    # preProcess = "pca",
                   trControl = control)

# random forest 
set.seed(123)
model_rf <- train(deposit~.,
                  data = dfTrain,
                  method = "rf",
                  ntree = 20,
                  tuneLength = 5,
                  trControl = control,
                  tuneGrid = rfGrid)
print(model_rf)
plot(model_rf)
# XGBoosting
set.seed(123)
model_xgb <- train(deposit~.,
                   data = dfTrain,
                   method = "xgbTree",
                   trControl = control,
                   tuneGrid = parameterGrid)

print(model_xgb)
plot(model_xgb)
model_xgb$bestTune
stopCluster(cl)

### actual prediction
# Logistic regression
pred_glm_raw <- predict.train(model_glm,
                              newdata = dfTest,
                              type = "raw") # use actual predictions

# Random forest
pred_rf_raw <- predict.train(model_rf,
                             newdata = dfTest,
                             type = "raw")

# XGBoost
pred_xgb_raw <- predict.train(model_xgb,
                              newdata = dfTest,
                              type = "raw")

### probabilities
# Logistic regression
pred_glm_prob <- predict.train(model_glm,
                               newdata = dfTest,
                               type = "prob") 
# Random forest
pred_rf_prob <- predict.train(model_rf,
                              newdata = dfTest,
                              type = "prob")

# XGBoost
pred_xgb_prob <- predict.train(model_xgb,
                               newdata = dfTest,
                               type = "prob")
### Confusion matrices for 3 methods 
confusionMatrix(data = pred_glm_raw,
                factor(dfTest$deposit),
                positive = "yes")

confusionMatrix(data = pred_rf_raw,
                factor(dfTest$deposit),
                positive = "yes")

confusionMatrix(data = pred_xgb_raw,
                factor(dfTest$deposit),
                positive = "yes")
### choose the final model from these three
model_list <- list(logistic_regression = model_glm,
                   random_forest = model_rf,
                   XGBoosting = model_xgb)

res <- resamples(model_list)
summary(res)

# Plot model results - Accuracy, AUC, and F1
# Three key performance metrics were chosen to compare model performance.
bwplot(res , metric = c("AUC", "F1"))

# Run a t-test to compare model performance (xbg and rf)
compare_models(model_rf, model_xgb)

varImpXGB <- varImp(model_xgb)
varImpXGB 
top4 <- varImpXGB$importance %>% head(n = 5)
top5
top5 <- rownames(top4)
top5[match('poutcomesuccess', top5)] <- "poutcome"
top5[match('contactunknown', top5)] <- "contact"
top5[match('housingyes', top5)] <- "housing"
top5[match('loanyes', top5)] <- "loan"
cbind(top5,top4)



