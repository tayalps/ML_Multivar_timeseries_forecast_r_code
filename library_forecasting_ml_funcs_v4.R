diskcleanup <- function(nm) {
  rm(list=nm, envir = .GlobalEnv); invisible( gc() )
}

decompose_date_feat <- function (df) {
  
  df$Year <- as.integer(format(df$Week, "%y"))
  df = df %>%
    mutate(Month = as.integer(month(Week)),
           Weeknum = as.integer(week(Week)),
           Day = as.integer(day(Week)),
    )
  
  df = df %>%
    mutate(quartofyear = as.integer(quarter(Week)),
           weekofmonth = as.integer(ceiling(day(Week) / 7))
    )
  return (df)
}

Create_lags <- function(df, start_lag, num_lags) {
  
  lags = seq(from = start_lag, to=start_lag+num_lags)
  lag_names <- paste("lag", formatC(lags, width = nchar(max(lags)), flag = "0"), 
                     sep = "_")
  lag_functions <- setNames(paste("dplyr::lag(., ", lags, ")"), lag_names)
  print(lag_functions)
  df = df %>%
    arrange(Customer, Product) %>%  
    group_by(Customer, Product) %>%
    mutate_at(vars(Sales.M), funs_(lag_functions)) 
  print(colnames(df))  
  return (df)
}

#some additional lag features
Create_additional_lag_feats <- function(df){
  
  lag_col_names = colnames(df[, (grepl( "lag" , names( df ))) ])
  i=1
  for (i in 1:(length(lag_col_names)-1) ) {
    span = i+1
    if( span <= length(lag_col_names)) {
      cols = lag_col_names[1:span]
      
      colname_sd = paste0("lagsd_", "1to", span, collapse = "")
      df[colname_sd] = rowSds(as.matrix(df[,cols]))
      
      colname_max = paste0("lagmax_", "1to", span, collapse = "")
      df[colname_max] = rowMaxs(as.matrix(df[,cols]))
    }
    
    colname_diff = paste0("lagdiff_", "1to", (i+1), collapse = "")
    df[colname_diff] = df[lag_col_names[1]] - df[lag_col_names[i+1]]
    
    colname_div = paste0("lagdiv_", "1to", (i+1), collapse = "")
    df[colname_div] = df[lag_col_names[1]] / df[lag_col_names[i+1]]
  }
  
  return (df)
}

Create_rolling_windows_means <- function(df, start_index, num_windows) {
  
  rollmean_l = seq(from = start_index, to=start_index+num_windows)
  rollmean_names <- paste("rollmean",formatC(rollmean_l, 
                                             width = nchar(max(rollmean_l)), flag = "0"), 
                          sep = "_")
  rollmean_functions <- setNames(
    paste('lag(roll_meanr(., ', rollmean_l, ")", ", 1)"), 
    rollmean_names)
  print(rollmean_functions)
  df = df %>%
    arrange(Customer, Product) %>%
    group_by(Customer, Product) %>%
    mutate_at(vars(Sales.M), funs_(rollmean_functions))
  print(colnames(df))
  return (df)
}

Create_rolling_windows_sd <- function(df, start_index, num_windows) {
  rollsd_l = seq(from = start_index, to=start_index+num_windows)
  rollsd_names <- paste("rollsd",formatC(rollsd_l, 
                                         width = nchar(max(rollsd_l)), flag = "0"), 
                        sep = "_")
  rollsd_functions <- setNames(
    paste('lag(roll_sdr(., ', rollsd_l, ")", ", 1)"), 
    rollsd_names)
  print(rollsd_functions)
  df = df %>%
    arrange(Customer, Product) %>%
    group_by(Customer, Product) %>%
    mutate_at(vars(Sales.M), funs_(rollsd_functions))
  print(colnames(df))
  return (df)
}

Create_AR_MA_feats <- function(df, start_index_lag=1, num_lags=4,
                               start_index_rollfeat = 2, num_rollfeat=8){
  
  df = df %>% 
    Create_lags(., start_index_lag, num_lags) %>%
    Create_rolling_windows_means(.,start_index_rollfeat,num_rollfeat) %>%
    Create_rolling_windows_sd(., start_index_rollfeat, num_rollfeat)
  
  return (df)
}

