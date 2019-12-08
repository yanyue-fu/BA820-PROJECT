library(readr)
library(tidyverse)
library(DT)
library(ggplot2)
library(data.table)
library(scales)
library(randomForest)
library(rpart) 
library(rpart.plot) 
library(gbm)


#load data + train and test 
bank <- fread(file="bankdata_cleaned.csv")
data <- bankdata[,c(-2,-3,-13)]
set.seed(820)
data$train <- sample(c(0,1), nrow(data), replace = T, prob = c(.5, .5)) 
bank_test <- data %>% filter(train == 0)
bank_train <- data %>% filter(train == 1)

#data preparation(Deposit)
f1 <- as.formula (Deposit~ age +            education  +    default    +     balance +      housing  +      
                    loan   +         duration  +   campaign +       pdays     +      previous   +     
                    job_admin    +   job_blue_collar  +job_entrepreneur + job_housemaid   + job_management + job_retired+ job_self_emply  +
                    job_services+ job_student   +job_technician +job_unemply +.data_unknown + 
                    m_divorced   + m_married   +  m_single )
x_train <- model.matrix(f1, bank_train)[, -1] 
y_train <- bank_train$Deposit
x_test <- model.matrix(f1, bank_test)[, -1] 
y_test <- bank_test$Deposit

#Random Forest
Deposit_rf <- randomForest(f1, bank_train, ntree = 500, do.trace=F) 
yhat_rf_train <- predict(Deposit_rf, bank_train)
mse_rf_train <- mean((yhat_rf_train - y_train) ^2) 
yhat_rf_test <- predict(Deposit_rf, bank_test)
mse_rf_test <- mean((yhat_rf_test - y_test) ^2) 
varImpPlot(Deposit_rf)
mean(data$duration)

paste("Random Forest Train MSE",mse_rf_train)
paste("Random Forest Train MSE",mse_rf_test)

data$DurationRange<-findInterval(data$duration, c(0,100,200,300,400,500,600,800,1000,1500,2000,2500,3000))
data1 <-data %>% group_by(DurationRange) %>% summarise(deposit=mean(Deposit))
ggplot(data1)+geom_col(aes(DurationRange,deposit))+scale_x_discrete(
                                                                    limits=c("0","100","200","300","400","500","600","800","1000","1500","2000","2500","3000"))


summary(data1)



#data preparation(balance)
f2 <- as.formula (balance  ~ age +            education  +    default    +    Deposit +      housing  +      
                    loan   +         duration  +   campaign +       pdays     +      previous   +     
                    job_admin    +   job_blue_collar  +job_entrepreneur + job_housemaid   + job_management + job_retired+ job_self_emply  +
                    job_services+ job_student   +job_technician +job_unemply +.data_unknown + 
                    m_divorced   + m_married   +  m_single )
x2_train <- model.matrix(f2, bank_train)[, -1] 
y2_train <- bank_train$balance
x2_test <- model.matrix(f2, bank_test)[, -1] 
y2_test <- bank_test$balance

#Random Forest
balance_rf <- randomForest(f2, bank_train, ntree = 500, do.trace=F) 
yhat2_rf_train <- predict(balance_rf, bank_train)
mse2_rf_train <- mean((yhat2_rf_train - y2_train) ^2) 
yhat2_rf_test <- predict(balance_rf, bank_test)
mse2_rf_test <- mean((yhat2_rf_test - y2_test) ^2) 
varImpPlot(balance_rf)

paste("Random Forest Train MSE",mse2_rf_train)
paste("Random Forest Train MSE",mse2_rf_test)

#data preparation(loan)
f3 <- as.formula (loan ~ age + education  +    default    +    Deposit +      housing  +      
                    balance   + duration  +   campaign +       pdays     +      previous   +     
                    job_admin    +   job_blue_collar  +job_entrepreneur + job_housemaid   + job_management + job_retired+ job_self_emply  +
                    job_services+ job_student   +job_technician +job_unemply +.data_unknown + 
                    m_divorced   + m_married   +  m_single )
x3_train <- model.matrix(f3, bank_train)[, -1] 
y3_train <- bank_train$loan
x3_test <- model.matrix(f3, bank_test)[, -1] 
y3_test <- bank_test$loan

#Random Forest
loan_rf <- randomForest(f3, bank_train, ntree = 500, do.trace=F) 
yhat3_rf_train <- predict(loan_rf, bank_train)
mse3_rf_train <- mean((yhat3_rf_train - y3_train) ^2) 
yhat3_rf_test <- predict(loan_rf, bank_test)
mse3_rf_test <- mean((yhat3_rf_test - y3_test) ^2) 
varImpPlot(loan_rf)

paste("Random Forest Train MSE",mse3_rf_train)
paste("Random Forest Train MSE",mse3_rf_test)

#data preparation(campaign)
f4 <- as.formula (campaign ~ age + education  +    default    +    Deposit +      housing  +      
                    balance   + duration  +   loan +       pdays     +      previous   +     
                    job_admin    +   job_blue_collar  +job_entrepreneur + job_housemaid   + job_management + job_retired+ job_self_emply  +
                    job_services+ job_student   +job_technician +job_unemply +.data_unknown + 
                    m_divorced   + m_married   +  m_single )
x4_train <- model.matrix(f4, bank_train)[, -1] 
y4_train <- bank_train$campaign
x4_test <- model.matrix(f3, bank_test)[, -1] 
y4_test <- bank_test$campaign

#Random Forest
campaign_rf <- randomForest(f4, bank_train, ntree = 500, do.trace=F) 
yhat4_rf_train <- predict(campaign_rf, bank_train)
mse4_rf_train <- mean((yhat4_rf_train - y4_train) ^2) 
yhat4_rf_test <- predict(campaign_rf, bank_test)
mse4_rf_test <- mean((yhat4_rf_test - y4_test) ^2) 
varImpPlot(campaign_rf)

paste("Random Forest Train MSE",mse4_rf_train)
paste("Random Forest Train MSE",mse4_rf_test)



