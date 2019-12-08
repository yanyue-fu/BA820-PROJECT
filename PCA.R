library(tidyverse)
library(factoextra)
library(corrplot)

## new packages
#install.packages("psych")
library(psych)

#install.packages("GPArotation")
library(GPArotation)
data<-read_csv("bankdata_cleaned.csv")
try<-data %>% select(is.numeric)

view(data)
barplot(table(data$marital))

S<-data %>% select(age:job_unemply)
S<-S %>% select(-job,-marital,-deposit)
view(S)
S$age<-scale(S$age)
S$balance<-scale(S$balance)
S$duration<-scale(S$duration)
View(S)

corrplot(cor(S),
         type="upper",
         method="number")

S_pca = prcomp(S, center=TRUE, scale = TRUE)

summary(S_pca)
S_pca$rotation
## fit the plot
fviz_screeplot(S_pca, addlabels=T, ylim =c(0,100))
fviz_pca_contrib(S_pca, choice="var", axes = 2)
fviz_pca_var(S_pca,
             col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE) 

fviz_screeplot(S_pca,
               addlabels = TRUE,
               ylim = c(0,100))
## get the eigenvalue
get_eigenvalue(S_pca)
S_pcs = predict(S_pca, newdata=S)  ## the numeric data
S_pcs = S_pcs[, 1:14]
head(S_pcs)

S_pcs=as.data.frame(S_pcs)

fviz_nbclust(S_pcs, kmeans, method = "silhouette", k.max=50)
fviz_nbclust(S_pcs, kmeans, method = "wss", k.max=50)

stock.cluster <- kmeans(S_pcs, centers = 7, iter.max = 25, nstart = 25)
fviz_cluster(stock.cluster, S_pcs)
