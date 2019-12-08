library(tidyverse)
library(factoextra)
library(corrplot)
library(psych)
library(GPArotation)
library(cluster)


bank <- read_csv(file = "bankdata_cleaned.csv")
bank_term_deposit <- bank %>% select(deposit)
data <- bank %>% select(., -job, -marital, -deposit)


### We can not run a pca because our data has 
### binary variables that are dummy and not continue
### Also there is no concern as 

#### Clustering
fviz_nbclust(data, kmeans, method = "silhouette", k.max=15)
## gives us 2

fviz_nbclust(data, kmeans, method = "wss", k.max=15)
## looks more like 3 for elbow group

k <- kmeans(data, centers=4, iter.max = 500, nstart = 500)
plot(silhouette(k$cluster, dist=dist(data)), col = 1:3, border = NA)
k$size


# H clustering

data_dist <- dist(data, method = "euclidean")
hfit <- hclust(data_dist, method = "ward.D")
plot(hfit)
hclust <- cutree(hfit, k=3)
rect.hclust(hfit, k = 3, border ="green")

data %>% count(hclust)

#### Clustering

##### KNN

data_knn <- data %>% select(-hclust)

smp_size <- floor(0.95 * nrow(data))
## set the seed to make your partition reproducible
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

data_train_knn <- data_knn[train_ind, ]
data_test_knn <- data_knn[-train_ind, ]

data_dist_knn <- dist(data_train_knn, method = "euclidean")
hfit_knn <- hclust(data_dist_knn, method = "ward.D")
plot(hfit_knn)
hclust_knn <- cutree(hfit_knn, k=3)
rect.hclust(hfit, k = 3, border ="green")


knn <- knn(train = data_train_knn, test = data_test_knn, cl = hclust_knn, k =101)
## utilized 101 as it was an odd number rounded from the sqrt of observations of the train

data_train_knn <- cbind(hclust_knn, data_train_knn)
data_test_knn <- cbind(knn, data_test_knn)
#### Knn

