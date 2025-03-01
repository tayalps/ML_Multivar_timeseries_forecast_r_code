###############
# Install the packages and load them
##############
rm(list=ls(all=T))
gc()

suppressPackageStartupMessages({
  list.of.packages <- c("stringr", "qdap", "tidytext", "tm", "caret",
                        "e1071", "ggplot2", "dplyr", "digest", "gbm",
                        "randomForest", "rpart", "rpart.plot", "Rtsne",
                        "TTR", "matrixStats", "zoo", "lubridate", "RcppRoll")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  
  lapply(list.of.packages, require, character.only = TRUE, )#load the libraries
})


library("catboost")
###############
# Define some global variables
##############
dirname ="C:/PraveenTayal/GL_Analytics_Pandit/timeseries_forecasting/walmart-recruiting-store-sales-forecasting/"
setwd(dirname)

start_index_lag=1
num_lags=8
start_index_rollfeat = 2 
num_rollfeat=10
forecast_horizon = 12 #12 weeks

###############
# load the function library 
##############
source(file = 'library_forecasting_ml_funcs.R')

###############
# Read the data
##############
filename = "train.csv"
train=read.csv(filename, header=T, na.strings=c("","NA"), stringsAsFactors = FALSE)
dim(train)

train$Week = as.Date(train$Week, format = '%m/%d/%Y') #convert date
train %>% glimpse()
summary(train)

#summary shows that the few records have sales in -ve
#that is unlikely. Hence removing such records
length(train$Sales.M[train$Sales.M < 0])
train$Sales.M[train$Sales.M < 0]= 0 

#Drop the cases when the group "product, customer" records are less than
#frequent. - padding with zero sale value is another option
min_period_len = 20 #in weeks
train = train %>%
  arrange(Customer, Product) %>%
  group_by(Customer, Product) %>%
  filter(n() > min_period_len)

dim(train) 

filename = "test.csv"
test=read.csv(filename, header=T, na.strings=c("","NA"), stringsAsFactors = FALSE)
test %>% glimpse()
df.total_master = test %>%
  mutate(Week = as.Date(Week, format = '%m/%d/%Y')) %>%
  filter(., Product %in% unique(train$Product)) %>%
  filter(., Customer %in% unique(train$Customer)) %>%
  bind_rows(train) %>%
  arrange(Week) %>%
  arrange(Customer, Product) %>%
  select(Week, Customer, Product, Sales.M, Holiday) %>%
  mutate(Holiday = as.integer(Holiday))


############
# Feature Engineering
###########
## decompose the date - create time based features
## add lag and moving avg and a few extra features
train = train %>% 
  mutate(Holiday = as.integer(Holiday)) %>%
  decompose_date_feat(.) %>%
  Create_AR_MA_feats(., start_index_lag,num_lags,
                   start_index_rollfeat,num_rollfeat) %>%
  Create_additional_lag_feats(.)

train = na.omit(train)
dim(train) 

#lets start with tree based model where one hot encoding
#for categorical types is not mandatory
drops <- c("Sales.M", "Week", "ï..Avg.Price", "Discount", "Attrition")
feat.names = colnames( train[ , !(names(train) %in% drops)])


pivot_date = "8/14/2017"

# Split the data into training and test sets
Xtrain <- train %>% filter(Week < as.Date(
  pivot_date, format = '%m/%d/%Y'))
Xval <- train %>% filter(Week >= as.Date(
  pivot_date, format = '%m/%d/%Y'))

dim(Xtrain); dim(Xval)
tra = Xtrain[,feat.names]
vra = Xval[,feat.names]

dval<-xgb.DMatrix(data=data.matrix(vra),label=(Xval$Sales.M))
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=(Xtrain$Sales.M) )


library(catboost)
set.seed(314)
# Parameters
dval <- catboost.load_pool(vra, label = Xval$Sales.M)
dtrain <- catboost.load_pool(tra, label = Xtrain$Sales.M)
dim(dtrain); dim(dval) 

params <- list(iterations = 2000,
               metric_period = 100,
               #       task_type = "GPU",
               loss_function = "RMSE",
               eval_metric = "RMSE",
               random_strength = 0.5,
               depth = 7,
               # early_stopping_rounds = 400,
               learning_rate = 0.1,
               l2_leaf_reg = 0.1,
               random_seed = 93)

# Fit
fit <- catboost.train(dtrain, dval, params = params)
# Variable importance plot
catboost.get_feature_importance(fit, dval) %>% 
  as.data.frame() %>% 
  setNames("Importance") %>% 
  tibble::rownames_to_column("Feature") %>% 
  arrange(desc(Importance)) %>%
  top_n(15) %>%
  ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip()

##Prediction of validation set
#transform the log to original form
pred1= catboost.predict(fit, dval)

d <- tibble(pred = pred1
            , obs = Xval$Sales.M) %>% 
  mutate(resid = pred - obs,
         resid_sq = resid^2)
sstot <- sum((d$pred - mean(d$obs))^2)
ssresid <- sum(d$resid_sq)
sprintf("percent variance explained, R^2: %1.1f%%", 
        100 * (1 - ssresid / sstot))

rmse= mean((pred1-Xval$Sales.M)^2)
rmse

plot(d$pred, d$obs, pch=16, col="blue", cex=0.75,
     xlab="Predicted Power Output",
     ylab="Observed Power Output",
     main="CATBOOST: Observed vs. Predicted")
lines(d$pred,
      lm(a~b, data=data.frame(a=d$obs, b=d$pred))$fitted,
      lwd=2, col="red")
plot(d$pred, d$resid, pch=16, col="blue", cex=0.75,
     ylab="Predicted Power Output",
     xlab="Residual Power Output",
     main="CatBoost: Residual vs. Predicted")
lines(d$pred,
      lm(a~b, data=data.frame(a=d$resid, b=d$pred))$fitted,
      lwd=2, col="red")


#arrange the prediction and actual together and write in csv
#file. this file can be loaded in Tableau or Shiny for better
#visualization and inspection.
prediction.tbl = data.frame(Product = Xval$Product,
                            Customer = Xval$Customer,
                            Sales.M=round(pred1,1), Week=Xval$Week) %>%
  arrange(Customer, Product) %>%  
  mutate(key = "Predictions")

output = train %>%
  select(Product, Customer, Week, Sales.M, Holiday) %>%
  mutate(key = "Actual") %>%
  bind_rows(prediction.tbl)

write.csv(output, "Pred-output_catboost.csv")

rm(Xval, Xtrain, tra, vra, d, output, prediction.tbl); gc()

#########################
##Forecasting
##Week wise forecasting
##Recursive
######################
Dates = seq(max(train$Week), by = "week", length.out = forecast_horizon)
Dates
df.total = df.total_master
i=1
for (i in 2:length(Dates)){
  
   df_test <- df.total %>%
    filter(Week <= Dates[i]) %>% #filter all historical data with week 1 by 1
    decompose_date_feat(.)
  
  df_test = df_test %>% 
    group_by(Customer, Product) %>% 
    filter(n() > 13) #to avoid items that are not enough in size
  
  #build the feature engg for the unseen week
  df_test =  df_test  %>%
    Create_AR_MA_feats(., start_index_lag,num_lags,
                       start_index_rollfeat,num_rollfeat) %>%
    Create_additional_lag_feats(.)
  
  #filter that unseen week features
  test <- df_test[df_test$Week == (Dates[i]),]
  
  #predict on that unseen week
  pred1= catboost.predict(fit, catboost.load_pool(test[,feat.names])) 
  test$Sales.M = round(pred1,2)
  
  test = test %>%
    select(Week, Customer, Product, Sales.M, Holiday) 
  
  #add this prediction to the data
  #this is recursion - now pred1 will be used to predict the next unseen week
  #in the loop
  df.total  = df.total %>%
    left_join(., test, by=c("Week", "Customer", "Product", "Holiday") ) %>%
    mutate(Sales.M.x = ifelse(is.na( Sales.M.x), Sales.M.y, Sales.M.x)) %>%
    rename(Sales.M = Sales.M.x) %>%
    select(-Sales.M.y)
  
  
  print(i)
  gc()
  
}

rm(df_test, test); gc()

##Combine the historical and forecast together
#for better analysis in Tableau or Shiny later on
forecast.tbl = df.total %>%
  filter(Week >= Dates[2]) %>%
  select(Product, Customer, Week, Sales.M, Holiday) %>%
  arrange(Customer, Product) %>% 
  mutate(key = "Forecast")
  
output = train %>%
  select(Product, Customer, Week, Sales.M, Holiday) %>%
  mutate(key = "Historical") %>%
  arrange(Customer, Product) %>% 
  bind_rows(forecast.tbl)

write.csv(output, "forecast_12_weeks_catboost.csv")

rm(train, df.total, df.total_master, forecast.tbl, output); gc()
