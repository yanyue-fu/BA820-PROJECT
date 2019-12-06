library(tidyverse)
library(factoextra)
library(corrplot)

## new packages
#install.packages("psych")
library(psych)

#install.packages("GPArotation")
library(GPArotation)
data<-read_csv("Documents/BA820/group/bankdata_cleaned.csv")
view(data)
barplot(table(data$marital))
S<-data %>% select(age:job_unemply)
S<-S %>% select(-job,-marital,-deposit)


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
S_pcs = S_pcs[, 1:11]
head(S_pcs)

S_pcs=as.data.frame(S_pcs)

fviz_nbclust(S_pcs, kmeans, method = "silhouette", k.max=50)
fviz_nbclust(S_pcs, kmeans, method = "wss", k.max=50)

stock.cluster <- kmeans(S_pcs, centers = 2, iter.max = 25, nstart = 25)
fviz_cluster(stock.cluster, S_pcs)
##############
d<-data %>% 
  select_if(is.numeric)

fviz_cluster(kw, w)
## copy the function from above
k_wss = function(k) {
  km = kmeans(w, k, nstart=25, iter=25)
  kwss = km$tot.withinss
  return(kwss)
}

## lets see what we get
x=1:20
wine_stats=map_dbl(x,k_wss)
plot(x,wine_stats,main="Wine K Optimization",type="b")

## silo eval
fviz_nbclust(w,kmeans,method="silhouette")

## plot the cluster
fviz_cluster(kmeans(w,2,25,25),w)
