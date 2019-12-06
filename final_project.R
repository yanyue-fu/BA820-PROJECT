library(tidyverse)


bank <- read_csv(file = "bankdata_cleaned.csv")
bank_term_deposit <- bank %>% select(deposit)
data <- bank %>% select(., -job, -marital, -deposit)




sum(is.na(data))
glimpse(data)
View(data)
