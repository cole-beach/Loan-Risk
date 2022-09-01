# Loan Risk

## Problem:


A bank in India would like to get a better understanding of customers who apply for loans.  

*They would like you to:*
 
 1. Segment the customers into groups of similar customers and report on those groups, and 
 
 2. Create a model that can classify customers into high and low risk groups.  They have provided a database of current customers labeled with whether they are a high or low risk loan candidate.
 
 ### Data notes:

**Income** = Indian Rupees

## Exploring & Cleaning the Dataset

```python
print(f'There are {df.duplicated().sum()} duplicated values','\n')
print(f'===========================================\n\n Missing Values:')
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
```
![image](https://user-images.githubusercontent.com/55251527/187971939-98d9e129-ba29-41e5-b796-8fff1c1ced01.png)
After dropping duplicated values we have 0 shown in the dataset

# Preprocessing

```python
# Scale
scaler = StandardScaler()

# Fit & Transfor the Dataset
scaled_df = scaler.fit_transform(df)
```
## KMeans Model

```python
%%time

# Locating the optimal number of clusters
k_range = range(2,10) # I'm using 11 to visualize the number 10 on the plot.
sils = []
inertias = []

# For Loop through the set range. 
for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(scaled_df)
  sils.append(silhouette_score(scaled_df, kmeans.labels_))
  inertias.append(kmeans.inertia_)

#plot inertias and silhouette scores for each number of clusters.
fig, axes = plt.subplots(1,2, figsize=(15,5))
axes[0].plot(k_range, sils)
axes[0].set_title('Silhouette Scores')
axes[0].set_xticks(k_range)
axes[1].plot(k_range, inertias)
axes[1].set_title('Inertia')
axes[1].set_xticks(k_range);
```

![image](https://user-images.githubusercontent.com/55251527/187972337-25799e09-a8ac-492a-a1a8-f73a19d3526e.png)

* Our Silhouette Score peaks at 6 clusters and there's a clear Inertia elbow at 6 clusters, which makes it safe to say the number of clusters we should use is 6. 

### Fitting a new model with the number of clusters.

```python
kmeans_6 = KMeans(n_clusters=6, random_state=42)
kmeans_6.fit(scaled_df)

# Adding the clusters as a column to the dataframe
df['clusters'] = kmeans_6.labels_

# Analyzing the Clusters
cluster_groups = df.groupby('clusters', as_index=False).mean()
cluster_groups
```

![image](https://user-images.githubusercontent.com/55251527/187972573-68bb0693-826a-4122-a167-7b2865364648.png)

### Visualization
![image](https://user-images.githubusercontent.com/55251527/187972614-7c580ef1-fb6f-429c-aee2-a1d3fa625c64.png)

#### Differences Between the Cluster Groups:

* Cluster-0 and Cluster-2 both have the most experience with just over 14-years. However, it's clear that the individuals in Cluster-0, although having that much experience, are not being compensated for the amount. It is very clear that those in Cluster-2 are receiving the highest income which can show that their amount of experince could play a roll  in their compensation. 

* Cluster-0 also receives less pay than Cluster-1 who only has a little over 4-years of experience and Cluster-3 who only have just over 10-years of experience. It even looks like Cluster-1 and Cluster-3 get paid the same amount. 

* From this information we can always dive further into other factors that can be playing into some Cluster groups making more with less experince. We can check to see if those groups have a higher number of married couples, rather than those that are on a single income. We can always look at the number of years that these gourps have been at their jobs. 


# Part II

## Loading the Model Dataset

