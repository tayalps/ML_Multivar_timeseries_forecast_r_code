###############
# Install the packages and load them
##############
rm(list=ls(all=T))
gc()

suppressPackageStartupMessages({
  list.of.packages <- c("stringr", "qdap", "tidytext", "tm", "caret",
                        "e1071", "ggplot2", "dplyr", "digest", "gbm",
                        "randomForest", "rpart", "rpart.plot", "Rtsne")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  
  lapply(list.of.packages, require, character.only = TRUE, )#load the libraries
})


#library("prophet")
library(TTR)
library(matrixStats)
library(zoo)
library(lubridate)
library(RcppRoll)
###############
# Define some global variables
##############
forecast_horizon = 1 # 1 week
dirname ="C:/PraveenTayal/GL_Analytics_Pandit/timeseries_forecasting/walmart-recruiting-store-sales-forecasting/"
setwd(dirname)

###############
# load the function library 
##############
source(file = 'library_forecasting_ml_funcs.R')

###############
# Read the data
##############


filename = "timeseries_data_annonymized.csv"
df=read.csv(filename, header=T, na.strings=c("","NA"), stringsAsFactors = FALSE)
dim(df)

df$Week = as.Date(df$Week, format = '%m/%d/%Y') #convert date
df %>% glimpse()
summary(df)

#summary shows that the few records have sales in -ve
#that is unlikely. Hence removing such records
length(df$Sales.M[df$Sales.M < 0]) #around 300 records found out of 4.2 lacs
df$Sales.M[df$Sales.M < 0]= 0 #remove negative sales 


#drop the cases when the group "product, customer" records are less than
#frequent. This may mean that there are customers that didn't buy specific 
#products. So either we can drop such in frequent customers or pad with zero sales.
min_period_len = 20 #in weeks - around 2 years
df = df %>%
  arrange(Customer, Product) %>%
  group_by(Customer, Product) %>%
  filter(n() > min_period_len)

dim(df) #after filter we have reduced the data from 4.2 - 3.9 lacs

filename = "test.csv"
test=read.csv(filename, header=T, na.strings=c("","NA"), stringsAsFactors = FALSE)
test %>% glimpse()
df.total = test %>%
  mutate(Week = as.Date(Week, format = '%m/%d/%Y')) %>%
  filter(., Product %in% unique(df$Product)) %>%
  filter(., Customer %in% unique(df$Customer)) %>%
  bind_rows(df) %>%
  arrange(Week) %>%
  arrange(Customer, Product) %>%
  select(Week, Customer, Product, Sales.M, Holiday) %>%
  mutate(Holiday = as.integer(Holiday))

#count_sales = df.total %>% group_by(Customer, Product) %>% summarise(count_sales = n())
#count_sales = count_sales$count_sales
#sort(count_sales)


############
# Feature Engineering
###########
## decompose the date - create time based features
df = df %>% 
  mutate(Holiday = as.integer(Holiday)) %>%
  decompose_date_feat(.)

##Create lag and sliding window variables -  very influential
df <- df[order(df$Week),] #arrange the data in increasing time series

start_index_lag=forecast_horizon
num_lags=4
start_index_rollfeat = forecast_horizon+1 
num_rollfeat=8

df = df %>%
  Create_AR_MA_feats(., start_index_lag,num_lags,
                     start_index_rollfeat,num_rollfeat) %>%
  Create_additional_lag_feats(.)

#Now the Target transformation
#sqrt, log, or lag diff - to avoid sudden peaks
df$Sales.M.log = log1p(df$Sales.M)

df = na.omit(df)
dim(df) # this leaves us with 3.7 lacs records
colnames(df)


#lets start with tree based model where one hot encoding
#for categorical types is not mandatory
drops <- c("Sales.M","Sales.M.log", "Week", "ï..Avg.Price", "Discount", "Attrition")
feat.names = colnames( df[ , !(names(df) %in% drops)])

train = df

pivot_date = "8/15/2016"

# Split the data into training and test sets
Xtrain <- train %>% filter(Week < as.Date(
  pivot_date, format = '%m/%d/%Y'))
Xval <- train %>% filter(Week >= as.Date(
  pivot_date, format = '%m/%d/%Y'))

dim(Xtrain); dim(Xval)
tra = Xtrain[,feat.names]
vra = Xval[,feat.names]

library(xgboost)
set.seed(314)

dval<-xgb.DMatrix(data=data.matrix(vra),label=log1p(Xval$Sales.M))
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=log1p(Xtrain$Sales.M) )
dim(dval); dim(dtrain)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.1, 
                max_depth           = 6,
                subsample           = 0.7,
                colsample_bytree    = 0.7
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500,
                    verbose             = 1,
                    early_stopping_rounds    = 200,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    eval=  'rmse'
)

#install.packages("Ckmeans.1d.dp")
library(Ckmeans.1d.dp)

##Feature importance
importance <- xgb.importance(
  feature_names = feat.names, model = clf)
print(importance)

xgb.ggplot.importance(importance_matrix = importance[1:15])
clf$best_score

##Prediction as per horizon
#transform the log to original form
pred1= exp(predict(clf, data.matrix(vra))) -1

d <- tibble(pred = pred1
            , obs = Xval$Sales.M) %>% 
  mutate(resid = pred - obs,
         resid_sq = resid^2)
sstot <- sum((d$pred - mean(d$obs))^2)
ssresid <- sum(d$resid_sq)
sprintf("percent variance explained, R^2: %1.1f%%", 
        100 * (1 - ssresid / sstot))

rmse= sqrt(mean((d$pred/d$obs-1)^2))

plot(d$pred, d$obs, pch=16, col="blue", cex=0.75,
     xlab="Predicted Power Output",
     ylab="Observed Power Output",
     main="XGBOOST: Observed vs. Predicted")
lines(d$pred,
      lm(a~b, data=data.frame(a=d$obs, b=d$pred))$fitted,
      lwd=2, col="red")



prediction.tbl = data.frame(Product = Xval$Product,
                            Customer = Xval$Customer,
                            Sales.M=round(pred1,1), Week=Xval$Week) %>%
  arrange(Customer, Product) %>%  
  mutate(label_text = str_glue("Date: {Week}
                               Sales: {Sales.M}")) %>%
  mutate(key = "Predictions")

dim(prediction.tbl)
dim(df)
output = df %>%
  select(Product, Customer, Week, Sales.M) %>%
  mutate(label_text = str_glue("Date: {Week}
                               Sales: {Sales.M}"),
         key = "Actual") %>%
  bind_rows(prediction.tbl)

write.csv(output, "1-week-pred-output.csv")


##Week wise forecasting
##recursive
Dates = seq(max(df$Week), by = "week", length.out = 8)
Dates
i=1
for (i in 1:length(Dates)){
  
   df_test <- df.total %>%
    filter(Week <= Dates[i]) %>% 
    decompose_date_feat(.)
  
  df_test = df_test %>% 
    group_by(Customer, Product) %>% 
    filter(n() > 13)
  
  df_test =  df_test  %>%
    Create_AR_MA_feats(., start_index_lag,num_lags,
                       start_index_rollfeat,num_rollfeat) %>%
    Create_additional_lag_feats(.)
  
  test <- df_test[df_test$Week == (Dates[i]),]
  
  pred2= exp(predict(clf, data.matrix(test[,feat.names]))) -1
  test$Sales.M = round(pred2,2)
  
  test = test %>%
    select(Week, Customer, Product, Sales.M, Holiday) 
  
  df.total  = df.total %>%
    left_join(., test, by=c("Week", "Customer", "Product", "Holiday") ) %>%
    mutate(Sales.M.x = ifelse(is.na( Sales.M.x), Sales.M.y, Sales.M.x)) %>%
    rename(Sales.M = Sales.M.x) %>%
    select(-Sales.M.y)
  
  
  print(i)
  gc()
  
}

write.csv(df.total, "forecast_6_weeks.csv")
