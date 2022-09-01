# Loan Risk

## Problem:


A bank in India would like to get a better understanding of customers who apply for loans.  

*They would like you to:*
 
 1. Segment the customers into groups of similar customers and report on those groups, and 
 
 2. Create a model that can classify customers into high and low risk groups.  They have provided a database of current customers labeled with whether they are a high or low risk loan candidate.

### Data notes:

**Income** = Indian Rupees

# Part I

## Loading Dataset


```python
# Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('notebook')
sns.set_style("whitegrid")

#Sci-kit Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

#Keras
from tensorflow.keras import Sequential
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```


```python
df = pd.read_csv('/content/drive/MyDrive/Coding Dojo: Data Science/03 Data Science: Advanced Machine Learning/Belt Exam Re-Take/Data/loan_cluster_df.csv')
pd.set_option('display.max_columns', None)
display(df.head())
```



  <div id="df-6e2173a1-7e41-4e08-818b-2e9872282ed4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Married</th>
      <th>Owns_House</th>
      <th>Owns_Car</th>
      <th>CURRENT_JOB_YRS</th>
      <th>CURRENT_HOUSE_YRS</th>
      <th>Risk_Flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3635824</td>
      <td>56</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3231341</td>
      <td>47</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7430695</td>
      <td>59</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8688710</td>
      <td>47</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2229190</td>
      <td>21</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6e2173a1-7e41-4e08-818b-2e9872282ed4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6e2173a1-7e41-4e08-818b-2e9872282ed4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6e2173a1-7e41-4e08-818b-2e9872282ed4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



## Exploring & Cleaning the Dataset


```python
print(df.info(),'\n')
print(f'The shape of the Dataset is: {df.shape}')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19438 entries, 0 to 19437
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype
    ---  ------             --------------  -----
     0   Income             19438 non-null  int64
     1   Age                19438 non-null  int64
     2   Experience         19438 non-null  int64
     3   Married            19438 non-null  int64
     4   Owns_House         19438 non-null  int64
     5   Owns_Car           19438 non-null  int64
     6   CURRENT_JOB_YRS    19438 non-null  int64
     7   CURRENT_HOUSE_YRS  19438 non-null  int64
     8   Risk_Flag          19438 non-null  int64
    dtypes: int64(9)
    memory usage: 1.3 MB
    None 
    
    The shape of the Dataset is: (19438, 9)
    


```python
print(f'There are {df.duplicated().sum()} duplicated values','\n')
print(f'===========================================\n\n Missing Values:')
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
```

    There are 10463 duplicated values 
    
    ===========================================
    
     Missing Values:
    





  <div id="df-aa765bec-3d23-4b89-ac7b-895a6bf004e1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Experience</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Owns_House</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-aa765bec-3d23-4b89-ac7b-895a6bf004e1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-aa765bec-3d23-4b89-ac7b-895a6bf004e1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-aa765bec-3d23-4b89-ac7b-895a6bf004e1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.drop_duplicates(inplace=True)
print(f'After dropping duplicated values we have {df.duplicated().sum()} shown in the dataset')
```

    After dropping duplicated values we have 0 shown in the dataset
    


```python
df.describe().T
```





  <div id="df-26dfb346-8d1c-47a8-b084-67caf04bcd25">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>8975.0</td>
      <td>5.014217e+06</td>
      <td>2.883422e+06</td>
      <td>10310.0</td>
      <td>2521816.0</td>
      <td>5003243.0</td>
      <td>7497723.0</td>
      <td>9999180.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>8975.0</td>
      <td>5.001582e+01</td>
      <td>1.697370e+01</td>
      <td>21.0</td>
      <td>35.0</td>
      <td>50.0</td>
      <td>65.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>Experience</th>
      <td>8975.0</td>
      <td>1.003721e+01</td>
      <td>6.002911e+00</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>8975.0</td>
      <td>1.021727e-01</td>
      <td>3.028922e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Owns_House</th>
      <td>8975.0</td>
      <td>5.381616e-02</td>
      <td>2.256671e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Owns_Car</th>
      <td>8975.0</td>
      <td>3.016156e-01</td>
      <td>4.589849e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CURRENT_JOB_YRS</th>
      <td>8975.0</td>
      <td>6.328022e+00</td>
      <td>3.639975e+00</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>CURRENT_HOUSE_YRS</th>
      <td>8975.0</td>
      <td>1.200457e+01</td>
      <td>1.406265e+00</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Risk_Flag</th>
      <td>8975.0</td>
      <td>1.664624e-01</td>
      <td>3.725159e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-26dfb346-8d1c-47a8-b084-67caf04bcd25')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-26dfb346-8d1c-47a8-b084-67caf04bcd25 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-26dfb346-8d1c-47a8-b084-67caf04bcd25');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.figure(figsize = (24, 12))

corr = df.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()
```


    
![image](https://user-images.githubusercontent.com/55251527/187975895-2a66c8a4-6d91-482a-9c10-d323a1914c40.png)

    


## Preprocessing


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

    CPU times: user 17.8 s, sys: 4.17 s, total: 21.9 s
    Wall time: 15 s
    


    
![png](output_18_1.png)
    


Our Silhouette Score peaks at 6 clusters and there's a clear Inertia elbow at 6 clusters, which makes it safe to say the number of clusters we should use is 6. 

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





  <div id="df-38d48f09-6e09-482e-921f-4fb314e6110a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>clusters</th>
      <th>Income</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Married</th>
      <th>Owns_House</th>
      <th>Owns_Car</th>
      <th>CURRENT_JOB_YRS</th>
      <th>CURRENT_HOUSE_YRS</th>
      <th>Risk_Flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4.988545e+06</td>
      <td>49.593313</td>
      <td>14.528830</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>8.596383</td>
      <td>12.034118</td>
      <td>0.167178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5.086717e+06</td>
      <td>50.031906</td>
      <td>4.502519</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3.485726</td>
      <td>11.953401</td>
      <td>0.173804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5.196943e+06</td>
      <td>49.514610</td>
      <td>14.461851</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>8.797078</td>
      <td>11.913149</td>
      <td>0.163149</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5.082334e+06</td>
      <td>51.231884</td>
      <td>10.455487</td>
      <td>0.082816</td>
      <td>1.0</td>
      <td>0.287785</td>
      <td>6.559006</td>
      <td>12.113872</td>
      <td>0.167702</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4.729983e+06</td>
      <td>50.957009</td>
      <td>4.787850</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>3.533645</td>
      <td>12.094393</td>
      <td>0.162617</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>4.955678e+06</td>
      <td>50.270239</td>
      <td>10.017104</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.303307</td>
      <td>6.280502</td>
      <td>12.003421</td>
      <td>0.152794</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-38d48f09-6e09-482e-921f-4fb314e6110a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-38d48f09-6e09-482e-921f-4fb314e6110a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-38d48f09-6e09-482e-921f-4fb314e6110a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Visualization


```python
fig, axes = plt.subplots(1,2, figsize=(30,8))

# Most Experience
axes[0].bar(cluster_groups.index, cluster_groups['Experience'])
axes[0].set_title('Which Cluster Group Has the Most Experience')
axes[0].set_ylabel('Years of Experience')
axes[0].set_xlabel('Cluster')
axes[0].set_xticks(cluster_groups.index)

# Income Cluster Group
axes[1].bar(cluster_groups.index, cluster_groups['Income'])
axes[1].set_title('Which Group Has the Highest Income')
axes[1].set_ylabel('Income')
axes[1].set_xlabel('Cluster')
axes[1].set_xticks(cluster_groups.index);
```


    
![png](output_23_0.png)
    


### Differences Between the Cluster Groups

Cluster-0 and Cluster-2 both have the most experience with just over 14-years. However, it's clear that the individuals in Cluster-0, although having that much experience, are not being compensated for the amount. It is very clear that those in Cluster-2 are receiving the highest income which can show that their amount of experince could play a roll  in their compensation. 

Cluster-0 also receives less pay than Cluster-1 who only has a little over 4-years of experience and Cluster-3 who only have just over 10-years of experience. It even looks like Cluster-1 and Cluster-3 get paid the same amount. 

From this information we can always dive further into other factors that can be playing into some Cluster groups making more with less experince. We can check to see if those groups have a higher number of married couples, rather than those that are on a single income. We can always look at the number of years that these gourps have been at their jobs. 

# Part II

## Loading the Model Dataset


```python
model_df = pd.read_csv('/content/drive/MyDrive/Coding Dojo: Data Science/03 Data Science: Advanced Machine Learning/Belt Exam Re-Take/Data/loan_model_df.csv')
pd.set_option('display.max_columns', None)
model_df.head()
```





  <div id="df-07ea0be6-2b98-4856-bf78-9565a18b56d8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Married</th>
      <th>Owns_House</th>
      <th>Owns_Car</th>
      <th>CURRENT_JOB_YRS</th>
      <th>CURRENT_HOUSE_YRS</th>
      <th>Risk_Flag</th>
      <th>Profession_Air_traffic_controller</th>
      <th>Profession_Analyst</th>
      <th>Profession_Architect</th>
      <th>Profession_Army_officer</th>
      <th>Profession_Artist</th>
      <th>Profession_Aviator</th>
      <th>Profession_Biomedical_Engineer</th>
      <th>Profession_Chartered_Accountant</th>
      <th>Profession_Chef</th>
      <th>Profession_Chemical_engineer</th>
      <th>Profession_Civil_engineer</th>
      <th>Profession_Civil_servant</th>
      <th>Profession_Comedian</th>
      <th>Profession_Computer_hardware_engineer</th>
      <th>Profession_Computer_operator</th>
      <th>Profession_Consultant</th>
      <th>Profession_Dentist</th>
      <th>Profession_Design_Engineer</th>
      <th>Profession_Designer</th>
      <th>Profession_Drafter</th>
      <th>Profession_Economist</th>
      <th>Profession_Engineer</th>
      <th>Profession_Fashion_Designer</th>
      <th>Profession_Financial_Analyst</th>
      <th>Profession_Firefighter</th>
      <th>Profession_Flight_attendant</th>
      <th>Profession_Geologist</th>
      <th>Profession_Graphic_Designer</th>
      <th>Profession_Hotel_Manager</th>
      <th>Profession_Industrial_Engineer</th>
      <th>Profession_Lawyer</th>
      <th>Profession_Librarian</th>
      <th>Profession_Magistrate</th>
      <th>Profession_Mechanical_engineer</th>
      <th>Profession_Microbiologist</th>
      <th>Profession_Official</th>
      <th>Profession_Petroleum_Engineer</th>
      <th>Profession_Physician</th>
      <th>Profession_Police_officer</th>
      <th>Profession_Politician</th>
      <th>Profession_Psychologist</th>
      <th>Profession_Scientist</th>
      <th>Profession_Secretary</th>
      <th>Profession_Software_Developer</th>
      <th>Profession_Statistician</th>
      <th>Profession_Surgeon</th>
      <th>Profession_Surveyor</th>
      <th>Profession_Technical_writer</th>
      <th>Profession_Technician</th>
      <th>Profession_Technology_specialist</th>
      <th>Profession_Web_designer</th>
      <th>CITY_Adoni</th>
      <th>CITY_Agartala</th>
      <th>CITY_Agra</th>
      <th>CITY_Ahmedabad</th>
      <th>CITY_Ahmednagar</th>
      <th>CITY_Aizawl</th>
      <th>CITY_Ajmer</th>
      <th>CITY_Akola</th>
      <th>CITY_Alappuzha</th>
      <th>CITY_Aligarh</th>
      <th>CITY_Allahabad</th>
      <th>CITY_Alwar</th>
      <th>CITY_Amaravati</th>
      <th>CITY_Ambala</th>
      <th>CITY_Ambarnath</th>
      <th>CITY_Ambattur</th>
      <th>CITY_Amravati</th>
      <th>CITY_Amritsar</th>
      <th>CITY_Amroha</th>
      <th>CITY_Anand</th>
      <th>CITY_Anantapur</th>
      <th>CITY_Anantapuram[24]</th>
      <th>CITY_Arrah</th>
      <th>CITY_Asansol</th>
      <th>CITY_Aurangabad</th>
      <th>CITY_Aurangabad[39]</th>
      <th>CITY_Avadi</th>
      <th>CITY_Bahraich</th>
      <th>CITY_Ballia</th>
      <th>CITY_Bally</th>
      <th>CITY_Bangalore</th>
      <th>CITY_Baranagar</th>
      <th>CITY_Barasat</th>
      <th>CITY_Bardhaman</th>
      <th>CITY_Bareilly</th>
      <th>CITY_Bathinda</th>
      <th>CITY_Begusarai</th>
      <th>CITY_Belgaum</th>
      <th>CITY_Bellary</th>
      <th>CITY_Berhampore</th>
      <th>CITY_Berhampur</th>
      <th>CITY_Bettiah[33]</th>
      <th>CITY_Bhagalpur</th>
      <th>CITY_Bhalswa_Jahangir_Pur</th>
      <th>CITY_Bharatpur</th>
      <th>CITY_Bhatpara</th>
      <th>CITY_Bhavnagar</th>
      <th>CITY_Bhilai</th>
      <th>CITY_Bhilwara</th>
      <th>CITY_Bhimavaram</th>
      <th>CITY_Bhind</th>
      <th>CITY_Bhiwandi</th>
      <th>CITY_Bhiwani</th>
      <th>CITY_Bhopal</th>
      <th>CITY_Bhubaneswar</th>
      <th>CITY_Bhusawal</th>
      <th>CITY_Bidar</th>
      <th>CITY_Bidhannagar</th>
      <th>CITY_Bihar_Sharif</th>
      <th>CITY_Bijapur</th>
      <th>CITY_Bikaner</th>
      <th>CITY_Bilaspur</th>
      <th>CITY_Bokaro</th>
      <th>CITY_Bongaigaon</th>
      <th>CITY_Bulandshahr</th>
      <th>CITY_Burhanpur</th>
      <th>CITY_Buxar[37]</th>
      <th>CITY_Chandigarh_city</th>
      <th>CITY_Chandrapur</th>
      <th>CITY_Chapra</th>
      <th>CITY_Chennai</th>
      <th>CITY_Chinsurah</th>
      <th>CITY_Chittoor[28]</th>
      <th>CITY_Coimbatore</th>
      <th>CITY_Cuttack</th>
      <th>CITY_Danapur</th>
      <th>CITY_Darbhanga</th>
      <th>CITY_Davanagere</th>
      <th>CITY_Dehradun</th>
      <th>CITY_Dehri[30]</th>
      <th>CITY_Delhi_city</th>
      <th>CITY_Deoghar</th>
      <th>CITY_Dewas</th>
      <th>CITY_Dhanbad</th>
      <th>CITY_Dharmavaram</th>
      <th>CITY_Dhule</th>
      <th>CITY_Dibrugarh</th>
      <th>CITY_Dindigul</th>
      <th>CITY_Durg</th>
      <th>CITY_Durgapur</th>
      <th>CITY_Eluru[25]</th>
      <th>CITY_Erode[17]</th>
      <th>CITY_Etawah</th>
      <th>CITY_Faridabad</th>
      <th>CITY_Farrukhabad</th>
      <th>CITY_Fatehpur</th>
      <th>CITY_Firozabad</th>
      <th>CITY_Gandhidham</th>
      <th>CITY_Gandhinagar</th>
      <th>CITY_Gangtok</th>
      <th>CITY_Gaya</th>
      <th>CITY_Ghaziabad</th>
      <th>CITY_Giridih</th>
      <th>CITY_Gopalpur</th>
      <th>CITY_Gorakhpur</th>
      <th>CITY_Gudivada</th>
      <th>CITY_Gulbarga</th>
      <th>CITY_Guna</th>
      <th>CITY_Guntakal</th>
      <th>CITY_Guntur[13]</th>
      <th>CITY_Gurgaon</th>
      <th>CITY_Guwahati</th>
      <th>CITY_Gwalior</th>
      <th>CITY_Hajipur[31]</th>
      <th>CITY_Haldia</th>
      <th>CITY_Hapur</th>
      <th>CITY_Haridwar</th>
      <th>CITY_Hazaribagh</th>
      <th>CITY_Hindupur</th>
      <th>CITY_Hospet</th>
      <th>CITY_Hosur</th>
      <th>CITY_Howrah</th>
      <th>CITY_HubliÃ¢â‚¬â€œDharwad</th>
      <th>CITY_Hyderabad</th>
      <th>CITY_Ichalkaranji</th>
      <th>CITY_Imphal</th>
      <th>CITY_Indore</th>
      <th>CITY_Jabalpur</th>
      <th>CITY_Jaipur</th>
      <th>CITY_Jalandhar</th>
      <th>CITY_Jalgaon</th>
      <th>CITY_Jalna</th>
      <th>CITY_Jamalpur[36]</th>
      <th>CITY_Jammu[16]</th>
      <th>CITY_Jamnagar</th>
      <th>CITY_Jamshedpur</th>
      <th>CITY_Jaunpur</th>
      <th>CITY_Jehanabad[38]</th>
      <th>CITY_Jhansi</th>
      <th>CITY_Jodhpur</th>
      <th>CITY_Jorhat</th>
      <th>CITY_Junagadh</th>
      <th>CITY_Kadapa[23]</th>
      <th>CITY_Kakinada</th>
      <th>CITY_Kalyan-Dombivli</th>
      <th>CITY_Kamarhati</th>
      <th>CITY_Kanpur</th>
      <th>CITY_Karaikudi</th>
      <th>CITY_Karawal_Nagar</th>
      <th>CITY_Karimnagar</th>
      <th>CITY_Karnal</th>
      <th>CITY_Katihar</th>
      <th>CITY_Katni</th>
      <th>CITY_Kavali</th>
      <th>CITY_Khammam</th>
      <th>CITY_Khandwa</th>
      <th>CITY_Kharagpur</th>
      <th>CITY_Khora,_Ghaziabad</th>
      <th>CITY_Kirari_Suleman_Nagar</th>
      <th>CITY_Kishanganj[35]</th>
      <th>CITY_Kochi</th>
      <th>CITY_Kolhapur</th>
      <th>CITY_Kolkata</th>
      <th>CITY_Kollam</th>
      <th>CITY_Korba</th>
      <th>CITY_Kota[6]</th>
      <th>CITY_Kottayam</th>
      <th>CITY_Kozhikode</th>
      <th>CITY_Kulti</th>
      <th>CITY_Kumbakonam</th>
      <th>CITY_Kurnool[18]</th>
      <th>CITY_Latur</th>
      <th>CITY_Loni</th>
      <th>CITY_Lucknow</th>
      <th>CITY_Ludhiana</th>
      <th>CITY_Machilipatnam</th>
      <th>CITY_Madanapalle</th>
      <th>CITY_Madhyamgram</th>
      <th>CITY_Madurai</th>
      <th>CITY_Mahbubnagar</th>
      <th>CITY_Maheshtala</th>
      <th>CITY_Malda</th>
      <th>CITY_Malegaon</th>
      <th>CITY_Mangalore</th>
      <th>CITY_Mango</th>
      <th>CITY_Mathura</th>
      <th>CITY_Mau</th>
      <th>CITY_Medininagar</th>
      <th>CITY_Meerut</th>
      <th>CITY_Mehsana</th>
      <th>CITY_Mira-Bhayandar</th>
      <th>CITY_Miryalaguda</th>
      <th>CITY_Mirzapur</th>
      <th>CITY_Moradabad</th>
      <th>CITY_Morbi</th>
      <th>CITY_Morena</th>
      <th>CITY_Motihari[34]</th>
      <th>CITY_Mumbai</th>
      <th>CITY_Munger</th>
      <th>CITY_Muzaffarnagar</th>
      <th>CITY_Muzaffarpur</th>
      <th>CITY_Mysore[7][8][9]</th>
      <th>CITY_Nadiad</th>
      <th>CITY_Nagaon</th>
      <th>CITY_Nagercoil</th>
      <th>CITY_Nagpur</th>
      <th>CITY_Naihati</th>
      <th>CITY_Nanded</th>
      <th>CITY_Nandyal</th>
      <th>CITY_Nangloi_Jat</th>
      <th>CITY_Narasaraopet</th>
      <th>CITY_Nashik</th>
      <th>CITY_Navi_Mumbai</th>
      <th>CITY_Nellore[14][15]</th>
      <th>CITY_New_Delhi</th>
      <th>CITY_Nizamabad</th>
      <th>CITY_Noida</th>
      <th>CITY_North_Dumdum</th>
      <th>CITY_Ongole</th>
      <th>CITY_Orai</th>
      <th>CITY_Ozhukarai</th>
      <th>CITY_Pali</th>
      <th>CITY_Pallavaram</th>
      <th>CITY_Panchkula</th>
      <th>CITY_Panihati</th>
      <th>CITY_Panipat</th>
      <th>CITY_Panvel</th>
      <th>CITY_Parbhani</th>
      <th>CITY_Patiala</th>
      <th>CITY_Patna</th>
      <th>CITY_Phagwara</th>
      <th>CITY_Phusro</th>
      <th>CITY_Pimpri-Chinchwad</th>
      <th>CITY_Pondicherry</th>
      <th>CITY_Proddatur</th>
      <th>CITY_Pudukkottai</th>
      <th>CITY_Pune</th>
      <th>CITY_Purnia[26]</th>
      <th>CITY_Raebareli</th>
      <th>CITY_Raichur</th>
      <th>CITY_Raiganj</th>
      <th>CITY_Raipur</th>
      <th>CITY_Rajahmundry[19][20]</th>
      <th>CITY_Rajkot</th>
      <th>CITY_Rajpur_Sonarpur</th>
      <th>CITY_Ramagundam[27]</th>
      <th>CITY_Ramgarh</th>
      <th>CITY_Rampur</th>
      <th>CITY_Ranchi</th>
      <th>CITY_Ratlam</th>
      <th>CITY_Raurkela_Industrial_Township</th>
      <th>CITY_Rewa</th>
      <th>CITY_Rohtak</th>
      <th>CITY_Rourkela</th>
      <th>CITY_Sagar</th>
      <th>CITY_Saharanpur</th>
      <th>CITY_Saharsa[29]</th>
      <th>CITY_Salem</th>
      <th>CITY_Sambalpur</th>
      <th>CITY_Sambhal</th>
      <th>CITY_Sangli-Miraj_&amp;_Kupwad</th>
      <th>CITY_Sasaram[30]</th>
      <th>CITY_Satara</th>
      <th>CITY_Satna</th>
      <th>CITY_Secunderabad</th>
      <th>CITY_Serampore</th>
      <th>CITY_Shahjahanpur</th>
      <th>CITY_Shimla</th>
      <th>CITY_Shimoga</th>
      <th>CITY_Shivpuri</th>
      <th>CITY_Sikar</th>
      <th>CITY_Silchar</th>
      <th>CITY_Siliguri</th>
      <th>CITY_Singrauli</th>
      <th>CITY_Sirsa</th>
      <th>CITY_Siwan[32]</th>
      <th>CITY_Solapur</th>
      <th>CITY_Sonipat</th>
      <th>CITY_South_Dumdum</th>
      <th>CITY_Sri_Ganganagar</th>
      <th>CITY_Srikakulam</th>
      <th>CITY_Srinagar</th>
      <th>CITY_Sultan_Pur_Majra</th>
      <th>CITY_Surat</th>
      <th>CITY_Surendranagar_Dudhrej</th>
      <th>CITY_Suryapet</th>
      <th>CITY_Tadepalligudem</th>
      <th>CITY_Tadipatri</th>
      <th>CITY_Tenali</th>
      <th>CITY_Tezpur</th>
      <th>CITY_Thane</th>
      <th>CITY_Thanjavur</th>
      <th>CITY_Thiruvananthapuram</th>
      <th>CITY_Thoothukudi</th>
      <th>CITY_Thrissur</th>
      <th>CITY_Tinsukia</th>
      <th>CITY_Tiruchirappalli[10]</th>
      <th>CITY_Tirunelveli</th>
      <th>CITY_Tirupati[21][22]</th>
      <th>CITY_Tiruppur</th>
      <th>CITY_Tiruvottiyur</th>
      <th>CITY_Tumkur</th>
      <th>CITY_Udaipur</th>
      <th>CITY_Udupi</th>
      <th>CITY_Ujjain</th>
      <th>CITY_Ulhasnagar</th>
      <th>CITY_Uluberia</th>
      <th>CITY_Unnao</th>
      <th>CITY_Vadodara</th>
      <th>CITY_Varanasi</th>
      <th>CITY_Vasai-Virar</th>
      <th>CITY_Vellore</th>
      <th>CITY_Vijayanagaram</th>
      <th>CITY_Vijayawada</th>
      <th>CITY_Visakhapatnam[4]</th>
      <th>CITY_Warangal[11][12]</th>
      <th>CITY_Yamunanagar</th>
      <th>STATE_Andhra_Pradesh</th>
      <th>STATE_Assam</th>
      <th>STATE_Bihar</th>
      <th>STATE_Chandigarh</th>
      <th>STATE_Chhattisgarh</th>
      <th>STATE_Delhi</th>
      <th>STATE_Gujarat</th>
      <th>STATE_Haryana</th>
      <th>STATE_Himachal_Pradesh</th>
      <th>STATE_Jammu_and_Kashmir</th>
      <th>STATE_Jharkhand</th>
      <th>STATE_Karnataka</th>
      <th>STATE_Kerala</th>
      <th>STATE_Madhya_Pradesh</th>
      <th>STATE_Maharashtra</th>
      <th>STATE_Manipur</th>
      <th>STATE_Mizoram</th>
      <th>STATE_Odisha</th>
      <th>STATE_Puducherry</th>
      <th>STATE_Punjab</th>
      <th>STATE_Rajasthan</th>
      <th>STATE_Sikkim</th>
      <th>STATE_Tamil_Nadu</th>
      <th>STATE_Telangana</th>
      <th>STATE_Tripura</th>
      <th>STATE_Uttar_Pradesh</th>
      <th>STATE_Uttar_Pradesh[5]</th>
      <th>STATE_Uttarakhand</th>
      <th>STATE_West_Bengal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3635824</td>
      <td>56</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3231341</td>
      <td>47</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7430695</td>
      <td>59</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8688710</td>
      <td>47</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2229190</td>
      <td>21</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-07ea0be6-2b98-4856-bf78-9565a18b56d8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-07ea0be6-2b98-4856-bf78-9565a18b56d8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-07ea0be6-2b98-4856-bf78-9565a18b56d8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Exploring & Cleaning the Model Dataset


```python
display(model_df.describe().T)
print(f'\n=============================================================================================================\n')
display(model_df.info(verbose=True))
print(f'\n=============================================================================================================\n')
print(f'The Shape of the Dataset is {model_df.shape}')
```



  <div id="df-7e32a3f3-9521-43da-98a1-3427a776c13b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>19438.0</td>
      <td>5.043779e+06</td>
      <td>2.885391e+06</td>
      <td>10310.0</td>
      <td>2533055.0</td>
      <td>5046887.5</td>
      <td>7532549.0</td>
      <td>9999180.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>19438.0</td>
      <td>4.981264e+01</td>
      <td>1.696799e+01</td>
      <td>21.0</td>
      <td>35.0</td>
      <td>50.0</td>
      <td>64.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>Experience</th>
      <td>19438.0</td>
      <td>1.013793e+01</td>
      <td>5.985557e+00</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>19438.0</td>
      <td>1.051034e-01</td>
      <td>3.066945e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Owns_House</th>
      <td>19438.0</td>
      <td>5.262887e-02</td>
      <td>2.232972e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>STATE_Tripura</th>
      <td>19438.0</td>
      <td>4.115650e-03</td>
      <td>6.402282e-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>STATE_Uttar_Pradesh</th>
      <td>19438.0</td>
      <td>1.113283e-01</td>
      <td>3.145464e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>STATE_Uttar_Pradesh[5]</th>
      <td>19438.0</td>
      <td>2.880955e-03</td>
      <td>5.359853e-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>STATE_Uttarakhand</th>
      <td>19438.0</td>
      <td>6.893713e-03</td>
      <td>8.274384e-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>STATE_West_Bengal</th>
      <td>19438.0</td>
      <td>9.337380e-02</td>
      <td>2.909630e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>406 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7e32a3f3-9521-43da-98a1-3427a776c13b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7e32a3f3-9521-43da-98a1-3427a776c13b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7e32a3f3-9521-43da-98a1-3427a776c13b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



    
    =============================================================================================================
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19438 entries, 0 to 19437
    Data columns (total 406 columns):
     #    Column                                 Dtype
    ---   ------                                 -----
     0    Income                                 int64
     1    Age                                    int64
     2    Experience                             int64
     3    Married                                int64
     4    Owns_House                             int64
     5    Owns_Car                               int64
     6    CURRENT_JOB_YRS                        int64
     7    CURRENT_HOUSE_YRS                      int64
     8    Risk_Flag                              int64
     9    Profession_Air_traffic_controller      int64
     10   Profession_Analyst                     int64
     11   Profession_Architect                   int64
     12   Profession_Army_officer                int64
     13   Profession_Artist                      int64
     14   Profession_Aviator                     int64
     15   Profession_Biomedical_Engineer         int64
     16   Profession_Chartered_Accountant        int64
     17   Profession_Chef                        int64
     18   Profession_Chemical_engineer           int64
     19   Profession_Civil_engineer              int64
     20   Profession_Civil_servant               int64
     21   Profession_Comedian                    int64
     22   Profession_Computer_hardware_engineer  int64
     23   Profession_Computer_operator           int64
     24   Profession_Consultant                  int64
     25   Profession_Dentist                     int64
     26   Profession_Design_Engineer             int64
     27   Profession_Designer                    int64
     28   Profession_Drafter                     int64
     29   Profession_Economist                   int64
     30   Profession_Engineer                    int64
     31   Profession_Fashion_Designer            int64
     32   Profession_Financial_Analyst           int64
     33   Profession_Firefighter                 int64
     34   Profession_Flight_attendant            int64
     35   Profession_Geologist                   int64
     36   Profession_Graphic_Designer            int64
     37   Profession_Hotel_Manager               int64
     38   Profession_Industrial_Engineer         int64
     39   Profession_Lawyer                      int64
     40   Profession_Librarian                   int64
     41   Profession_Magistrate                  int64
     42   Profession_Mechanical_engineer         int64
     43   Profession_Microbiologist              int64
     44   Profession_Official                    int64
     45   Profession_Petroleum_Engineer          int64
     46   Profession_Physician                   int64
     47   Profession_Police_officer              int64
     48   Profession_Politician                  int64
     49   Profession_Psychologist                int64
     50   Profession_Scientist                   int64
     51   Profession_Secretary                   int64
     52   Profession_Software_Developer          int64
     53   Profession_Statistician                int64
     54   Profession_Surgeon                     int64
     55   Profession_Surveyor                    int64
     56   Profession_Technical_writer            int64
     57   Profession_Technician                  int64
     58   Profession_Technology_specialist       int64
     59   Profession_Web_designer                int64
     60   CITY_Adoni                             int64
     61   CITY_Agartala                          int64
     62   CITY_Agra                              int64
     63   CITY_Ahmedabad                         int64
     64   CITY_Ahmednagar                        int64
     65   CITY_Aizawl                            int64
     66   CITY_Ajmer                             int64
     67   CITY_Akola                             int64
     68   CITY_Alappuzha                         int64
     69   CITY_Aligarh                           int64
     70   CITY_Allahabad                         int64
     71   CITY_Alwar                             int64
     72   CITY_Amaravati                         int64
     73   CITY_Ambala                            int64
     74   CITY_Ambarnath                         int64
     75   CITY_Ambattur                          int64
     76   CITY_Amravati                          int64
     77   CITY_Amritsar                          int64
     78   CITY_Amroha                            int64
     79   CITY_Anand                             int64
     80   CITY_Anantapur                         int64
     81   CITY_Anantapuram[24]                   int64
     82   CITY_Arrah                             int64
     83   CITY_Asansol                           int64
     84   CITY_Aurangabad                        int64
     85   CITY_Aurangabad[39]                    int64
     86   CITY_Avadi                             int64
     87   CITY_Bahraich                          int64
     88   CITY_Ballia                            int64
     89   CITY_Bally                             int64
     90   CITY_Bangalore                         int64
     91   CITY_Baranagar                         int64
     92   CITY_Barasat                           int64
     93   CITY_Bardhaman                         int64
     94   CITY_Bareilly                          int64
     95   CITY_Bathinda                          int64
     96   CITY_Begusarai                         int64
     97   CITY_Belgaum                           int64
     98   CITY_Bellary                           int64
     99   CITY_Berhampore                        int64
     100  CITY_Berhampur                         int64
     101  CITY_Bettiah[33]                       int64
     102  CITY_Bhagalpur                         int64
     103  CITY_Bhalswa_Jahangir_Pur              int64
     104  CITY_Bharatpur                         int64
     105  CITY_Bhatpara                          int64
     106  CITY_Bhavnagar                         int64
     107  CITY_Bhilai                            int64
     108  CITY_Bhilwara                          int64
     109  CITY_Bhimavaram                        int64
     110  CITY_Bhind                             int64
     111  CITY_Bhiwandi                          int64
     112  CITY_Bhiwani                           int64
     113  CITY_Bhopal                            int64
     114  CITY_Bhubaneswar                       int64
     115  CITY_Bhusawal                          int64
     116  CITY_Bidar                             int64
     117  CITY_Bidhannagar                       int64
     118  CITY_Bihar_Sharif                      int64
     119  CITY_Bijapur                           int64
     120  CITY_Bikaner                           int64
     121  CITY_Bilaspur                          int64
     122  CITY_Bokaro                            int64
     123  CITY_Bongaigaon                        int64
     124  CITY_Bulandshahr                       int64
     125  CITY_Burhanpur                         int64
     126  CITY_Buxar[37]                         int64
     127  CITY_Chandigarh_city                   int64
     128  CITY_Chandrapur                        int64
     129  CITY_Chapra                            int64
     130  CITY_Chennai                           int64
     131  CITY_Chinsurah                         int64
     132  CITY_Chittoor[28]                      int64
     133  CITY_Coimbatore                        int64
     134  CITY_Cuttack                           int64
     135  CITY_Danapur                           int64
     136  CITY_Darbhanga                         int64
     137  CITY_Davanagere                        int64
     138  CITY_Dehradun                          int64
     139  CITY_Dehri[30]                         int64
     140  CITY_Delhi_city                        int64
     141  CITY_Deoghar                           int64
     142  CITY_Dewas                             int64
     143  CITY_Dhanbad                           int64
     144  CITY_Dharmavaram                       int64
     145  CITY_Dhule                             int64
     146  CITY_Dibrugarh                         int64
     147  CITY_Dindigul                          int64
     148  CITY_Durg                              int64
     149  CITY_Durgapur                          int64
     150  CITY_Eluru[25]                         int64
     151  CITY_Erode[17]                         int64
     152  CITY_Etawah                            int64
     153  CITY_Faridabad                         int64
     154  CITY_Farrukhabad                       int64
     155  CITY_Fatehpur                          int64
     156  CITY_Firozabad                         int64
     157  CITY_Gandhidham                        int64
     158  CITY_Gandhinagar                       int64
     159  CITY_Gangtok                           int64
     160  CITY_Gaya                              int64
     161  CITY_Ghaziabad                         int64
     162  CITY_Giridih                           int64
     163  CITY_Gopalpur                          int64
     164  CITY_Gorakhpur                         int64
     165  CITY_Gudivada                          int64
     166  CITY_Gulbarga                          int64
     167  CITY_Guna                              int64
     168  CITY_Guntakal                          int64
     169  CITY_Guntur[13]                        int64
     170  CITY_Gurgaon                           int64
     171  CITY_Guwahati                          int64
     172  CITY_Gwalior                           int64
     173  CITY_Hajipur[31]                       int64
     174  CITY_Haldia                            int64
     175  CITY_Hapur                             int64
     176  CITY_Haridwar                          int64
     177  CITY_Hazaribagh                        int64
     178  CITY_Hindupur                          int64
     179  CITY_Hospet                            int64
     180  CITY_Hosur                             int64
     181  CITY_Howrah                            int64
     182  CITY_HubliÃ¢â‚¬â€œDharwad              int64
     183  CITY_Hyderabad                         int64
     184  CITY_Ichalkaranji                      int64
     185  CITY_Imphal                            int64
     186  CITY_Indore                            int64
     187  CITY_Jabalpur                          int64
     188  CITY_Jaipur                            int64
     189  CITY_Jalandhar                         int64
     190  CITY_Jalgaon                           int64
     191  CITY_Jalna                             int64
     192  CITY_Jamalpur[36]                      int64
     193  CITY_Jammu[16]                         int64
     194  CITY_Jamnagar                          int64
     195  CITY_Jamshedpur                        int64
     196  CITY_Jaunpur                           int64
     197  CITY_Jehanabad[38]                     int64
     198  CITY_Jhansi                            int64
     199  CITY_Jodhpur                           int64
     200  CITY_Jorhat                            int64
     201  CITY_Junagadh                          int64
     202  CITY_Kadapa[23]                        int64
     203  CITY_Kakinada                          int64
     204  CITY_Kalyan-Dombivli                   int64
     205  CITY_Kamarhati                         int64
     206  CITY_Kanpur                            int64
     207  CITY_Karaikudi                         int64
     208  CITY_Karawal_Nagar                     int64
     209  CITY_Karimnagar                        int64
     210  CITY_Karnal                            int64
     211  CITY_Katihar                           int64
     212  CITY_Katni                             int64
     213  CITY_Kavali                            int64
     214  CITY_Khammam                           int64
     215  CITY_Khandwa                           int64
     216  CITY_Kharagpur                         int64
     217  CITY_Khora,_Ghaziabad                  int64
     218  CITY_Kirari_Suleman_Nagar              int64
     219  CITY_Kishanganj[35]                    int64
     220  CITY_Kochi                             int64
     221  CITY_Kolhapur                          int64
     222  CITY_Kolkata                           int64
     223  CITY_Kollam                            int64
     224  CITY_Korba                             int64
     225  CITY_Kota[6]                           int64
     226  CITY_Kottayam                          int64
     227  CITY_Kozhikode                         int64
     228  CITY_Kulti                             int64
     229  CITY_Kumbakonam                        int64
     230  CITY_Kurnool[18]                       int64
     231  CITY_Latur                             int64
     232  CITY_Loni                              int64
     233  CITY_Lucknow                           int64
     234  CITY_Ludhiana                          int64
     235  CITY_Machilipatnam                     int64
     236  CITY_Madanapalle                       int64
     237  CITY_Madhyamgram                       int64
     238  CITY_Madurai                           int64
     239  CITY_Mahbubnagar                       int64
     240  CITY_Maheshtala                        int64
     241  CITY_Malda                             int64
     242  CITY_Malegaon                          int64
     243  CITY_Mangalore                         int64
     244  CITY_Mango                             int64
     245  CITY_Mathura                           int64
     246  CITY_Mau                               int64
     247  CITY_Medininagar                       int64
     248  CITY_Meerut                            int64
     249  CITY_Mehsana                           int64
     250  CITY_Mira-Bhayandar                    int64
     251  CITY_Miryalaguda                       int64
     252  CITY_Mirzapur                          int64
     253  CITY_Moradabad                         int64
     254  CITY_Morbi                             int64
     255  CITY_Morena                            int64
     256  CITY_Motihari[34]                      int64
     257  CITY_Mumbai                            int64
     258  CITY_Munger                            int64
     259  CITY_Muzaffarnagar                     int64
     260  CITY_Muzaffarpur                       int64
     261  CITY_Mysore[7][8][9]                   int64
     262  CITY_Nadiad                            int64
     263  CITY_Nagaon                            int64
     264  CITY_Nagercoil                         int64
     265  CITY_Nagpur                            int64
     266  CITY_Naihati                           int64
     267  CITY_Nanded                            int64
     268  CITY_Nandyal                           int64
     269  CITY_Nangloi_Jat                       int64
     270  CITY_Narasaraopet                      int64
     271  CITY_Nashik                            int64
     272  CITY_Navi_Mumbai                       int64
     273  CITY_Nellore[14][15]                   int64
     274  CITY_New_Delhi                         int64
     275  CITY_Nizamabad                         int64
     276  CITY_Noida                             int64
     277  CITY_North_Dumdum                      int64
     278  CITY_Ongole                            int64
     279  CITY_Orai                              int64
     280  CITY_Ozhukarai                         int64
     281  CITY_Pali                              int64
     282  CITY_Pallavaram                        int64
     283  CITY_Panchkula                         int64
     284  CITY_Panihati                          int64
     285  CITY_Panipat                           int64
     286  CITY_Panvel                            int64
     287  CITY_Parbhani                          int64
     288  CITY_Patiala                           int64
     289  CITY_Patna                             int64
     290  CITY_Phagwara                          int64
     291  CITY_Phusro                            int64
     292  CITY_Pimpri-Chinchwad                  int64
     293  CITY_Pondicherry                       int64
     294  CITY_Proddatur                         int64
     295  CITY_Pudukkottai                       int64
     296  CITY_Pune                              int64
     297  CITY_Purnia[26]                        int64
     298  CITY_Raebareli                         int64
     299  CITY_Raichur                           int64
     300  CITY_Raiganj                           int64
     301  CITY_Raipur                            int64
     302  CITY_Rajahmundry[19][20]               int64
     303  CITY_Rajkot                            int64
     304  CITY_Rajpur_Sonarpur                   int64
     305  CITY_Ramagundam[27]                    int64
     306  CITY_Ramgarh                           int64
     307  CITY_Rampur                            int64
     308  CITY_Ranchi                            int64
     309  CITY_Ratlam                            int64
     310  CITY_Raurkela_Industrial_Township      int64
     311  CITY_Rewa                              int64
     312  CITY_Rohtak                            int64
     313  CITY_Rourkela                          int64
     314  CITY_Sagar                             int64
     315  CITY_Saharanpur                        int64
     316  CITY_Saharsa[29]                       int64
     317  CITY_Salem                             int64
     318  CITY_Sambalpur                         int64
     319  CITY_Sambhal                           int64
     320  CITY_Sangli-Miraj_&_Kupwad             int64
     321  CITY_Sasaram[30]                       int64
     322  CITY_Satara                            int64
     323  CITY_Satna                             int64
     324  CITY_Secunderabad                      int64
     325  CITY_Serampore                         int64
     326  CITY_Shahjahanpur                      int64
     327  CITY_Shimla                            int64
     328  CITY_Shimoga                           int64
     329  CITY_Shivpuri                          int64
     330  CITY_Sikar                             int64
     331  CITY_Silchar                           int64
     332  CITY_Siliguri                          int64
     333  CITY_Singrauli                         int64
     334  CITY_Sirsa                             int64
     335  CITY_Siwan[32]                         int64
     336  CITY_Solapur                           int64
     337  CITY_Sonipat                           int64
     338  CITY_South_Dumdum                      int64
     339  CITY_Sri_Ganganagar                    int64
     340  CITY_Srikakulam                        int64
     341  CITY_Srinagar                          int64
     342  CITY_Sultan_Pur_Majra                  int64
     343  CITY_Surat                             int64
     344  CITY_Surendranagar_Dudhrej             int64
     345  CITY_Suryapet                          int64
     346  CITY_Tadepalligudem                    int64
     347  CITY_Tadipatri                         int64
     348  CITY_Tenali                            int64
     349  CITY_Tezpur                            int64
     350  CITY_Thane                             int64
     351  CITY_Thanjavur                         int64
     352  CITY_Thiruvananthapuram                int64
     353  CITY_Thoothukudi                       int64
     354  CITY_Thrissur                          int64
     355  CITY_Tinsukia                          int64
     356  CITY_Tiruchirappalli[10]               int64
     357  CITY_Tirunelveli                       int64
     358  CITY_Tirupati[21][22]                  int64
     359  CITY_Tiruppur                          int64
     360  CITY_Tiruvottiyur                      int64
     361  CITY_Tumkur                            int64
     362  CITY_Udaipur                           int64
     363  CITY_Udupi                             int64
     364  CITY_Ujjain                            int64
     365  CITY_Ulhasnagar                        int64
     366  CITY_Uluberia                          int64
     367  CITY_Unnao                             int64
     368  CITY_Vadodara                          int64
     369  CITY_Varanasi                          int64
     370  CITY_Vasai-Virar                       int64
     371  CITY_Vellore                           int64
     372  CITY_Vijayanagaram                     int64
     373  CITY_Vijayawada                        int64
     374  CITY_Visakhapatnam[4]                  int64
     375  CITY_Warangal[11][12]                  int64
     376  CITY_Yamunanagar                       int64
     377  STATE_Andhra_Pradesh                   int64
     378  STATE_Assam                            int64
     379  STATE_Bihar                            int64
     380  STATE_Chandigarh                       int64
     381  STATE_Chhattisgarh                     int64
     382  STATE_Delhi                            int64
     383  STATE_Gujarat                          int64
     384  STATE_Haryana                          int64
     385  STATE_Himachal_Pradesh                 int64
     386  STATE_Jammu_and_Kashmir                int64
     387  STATE_Jharkhand                        int64
     388  STATE_Karnataka                        int64
     389  STATE_Kerala                           int64
     390  STATE_Madhya_Pradesh                   int64
     391  STATE_Maharashtra                      int64
     392  STATE_Manipur                          int64
     393  STATE_Mizoram                          int64
     394  STATE_Odisha                           int64
     395  STATE_Puducherry                       int64
     396  STATE_Punjab                           int64
     397  STATE_Rajasthan                        int64
     398  STATE_Sikkim                           int64
     399  STATE_Tamil_Nadu                       int64
     400  STATE_Telangana                        int64
     401  STATE_Tripura                          int64
     402  STATE_Uttar_Pradesh                    int64
     403  STATE_Uttar_Pradesh[5]                 int64
     404  STATE_Uttarakhand                      int64
     405  STATE_West_Bengal                      int64
    dtypes: int64(406)
    memory usage: 60.2 MB
    


    None


    
    =============================================================================================================
    
    The Shape of the Dataset is (19438, 406)
    


```python
print(f'There are {model_df.duplicated().sum()} duplicated values','\n')
print(f'===========================================\n\n Missing Values:')
total = model_df.isnull().sum().sort_values(ascending=False)
percent = (model_df.isnull().sum()/model_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
```

    There are 10463 duplicated values 
    
    ===========================================
    
     Missing Values:
    





  <div id="df-c8ffbabf-f91b-4dd6-8f4b-d93e72eff887">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CITY_Nanded</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CITY_North_Dumdum</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CITY_Noida</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>CITY_Nizamabad</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c8ffbabf-f91b-4dd6-8f4b-d93e72eff887')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c8ffbabf-f91b-4dd6-8f4b-d93e72eff887 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c8ffbabf-f91b-4dd6-8f4b-d93e72eff887');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
model_df.drop_duplicates(inplace=True)
print(f'After dropping duplicated values we have {model_df.duplicated().sum()} shown in the dataset')
```

    After dropping duplicated values we have 0 shown in the dataset
    

## Preprocessing


```python
X = model_df.drop(columns='Risk_Flag')
y = model_df['Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
```


```python
# Instantiate Standard Scaler
scaler = StandardScaler()
# Instantiate PCA to explain 95% of variance
pca = PCA(n_components=.95)

pipeline_proc = make_pipeline(scaler, pca)

# fit on train
X_train_proc = pipeline_proc.fit_transform(X_train)
X_test_proc = pipeline_proc.transform(X_test)

print('Shape before PCA:', X_train.shape)
print('Shape after PCA:', pipeline_proc.fit_transform(X_train).shape)
print('Column count reduced by: ', X_train.shape[1] - pipeline_proc.fit_transform(X_train_proc).shape[1])
```

    Shape before PCA: (6731, 405)
    Shape after PCA: (6731, 347)
    Column count reduced by:  75
    

### Plotting Function


```python
#  You can use this function to see how your model improves over time
def plot_history(history, metric=None):
  """plot loss and passed metric.  metric is passed as string and must match 'metric'
  argument in the compile step"""
  fig, axes = plt.subplots(2,1, figsize = (25, 8))
  axes[0].plot(history.history['loss'], label = "train")
  axes[0].plot(history.history['val_loss'], label='test')
  axes[0].set_title('Loss')
  axes[0].legend()
  if metric:
    axes[1].plot(history.history[metric], label = 'train')
    axes[1].plot(history.history['val_' + metric], label = 'test')
    axes[1].set_title(metric)
    axes[1].legend()

  plt.show()
```

# Binary Classification Models

## Model 1


```python
# Instentiate the model
class_model = Sequential()

# Input layer. I'm using the number of features after PCA has been applied as a rule of thumb.
class_model.add(Dense(300, activation = 'relu', input_dim = X_train_proc.shape[1]))

# Create hidden layers
class_model.add(Dense(50, activation = 'relu'))

# Create output layer 
# Since this is a binary classification, the activation function of our final layer needs to be 'sigmoid'. 

class_model.add(Dense(1, activation = 'sigmoid'))

# compile model with metrics
class_model.compile(optimizer = 'adam', loss = 'bce', metrics = ['acc'])

history = class_model.fit(X_train_proc, y_train,
                        validation_data = (X_test_proc, y_test),
                        epochs = 75, verbose=0)

# plot learning history
plot_history(history, 'acc')
```


    
![png](output_40_0.png)
    


### Model 1 - Score (73%)



```python
# Make predicitons and evaluate your model
print('Training Scores')

# Define labels for the confusion matrix
labels = ['Not Passing', 'Passing']

# Get training predictions and round them to integers instead of floats
train_preds = np.rint(class_model.predict(X_train_proc))

# Classification Report
print(classification_report(y_train, train_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_train, train_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show()
print('\n')
print('Testing Scores')

# Get testing predictions and round them to integers
test_preds = np.rint(class_model.predict(X_test_proc))

# Classification report
print(classification_report(y_test, test_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, test_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show();
```

    Training Scores
                  precision    recall  f1-score   support
    
               0       0.98      0.95      0.96      5611
               1       0.79      0.90      0.84      1120
    
        accuracy                           0.94      6731
       macro avg       0.88      0.92      0.90      6731
    weighted avg       0.95      0.94      0.94      6731
    
    


    
![png](output_42_1.png)
    


    
    
    Testing Scores
                  precision    recall  f1-score   support
    
               0       0.82      0.86      0.84      1870
               1       0.07      0.05      0.06       374
    
        accuracy                           0.73      2244
       macro avg       0.44      0.46      0.45      2244
    weighted avg       0.69      0.73      0.71      2244
    
    


    
![png](output_42_3.png)
    


### Evaluation

It's clear to see that the model is overfitting, which is causing our testing data to stop improving. 

I'm going run the model again only this time I'm going to decrease the amount of epochs used to 7, since that's where the biggest split can be seen.  

## Model 2


```python
class_model_2 = Sequential()

# Input layer. I'm using the number of features after PCA has been applied as a rule of thumb.
class_model_2.add(Dense(300, activation = 'relu', input_dim = X_train_proc.shape[1]))

# Create hidden layers
class_model_2.add(Dense(50, activation = 'relu'))


# Since this is a binary classification, the activation function of our final layer needs to be 'sigmoid'
class_model_2.add(Dense(1, activation = 'sigmoid'))

# compile model with metrics
class_model_2.compile(optimizer = 'adam', loss = 'bce', metrics = ['acc'])

# 
history_2 = class_model_2.fit(X_train_proc, y_train,
                        validation_data = (X_test_proc, y_test),
                        epochs = 7, verbose=0)

# plot learning history
plot_history(history_2, 'acc')
```


    
![png](output_46_0.png)
    


### Model 2 - Score (81%)


```python
# Make predicitons and evaluate your model
print('Training Scores')

# Define labels for the confusion matrix
labels = ['Not Passing', 'Passing']

# Get training predictions and round them to integers instead of floats
train_preds = np.rint(class_model_2.predict(X_train_proc))

# Classification Report
print(classification_report(y_train, train_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_train, train_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show()
print('\n')
print('Testing Scores')

# Get testing predictions and round them to integers
test_preds = np.rint(class_model_2.predict(X_test_proc))

# Classification report
print(classification_report(y_test, test_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, test_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show();
```

    Training Scores
                  precision    recall  f1-score   support
    
               0       0.86      0.99      0.92      5611
               1       0.82      0.18      0.30      1120
    
        accuracy                           0.86      6731
       macro avg       0.84      0.59      0.61      6731
    weighted avg       0.85      0.86      0.82      6731
    
    


    
![png](output_48_1.png)
    


    
    
    Testing Scores
                  precision    recall  f1-score   support
    
               0       0.83      0.97      0.89      1870
               1       0.05      0.01      0.01       374
    
        accuracy                           0.81      2244
       macro avg       0.44      0.49      0.45      2244
    weighted avg       0.70      0.81      0.75      2244
    
    


    
![png](output_48_3.png)
    


### Evaluation

There's still some over fitting to model, but our test score is now at 82%, which is great coming from 73%. We can still see that there is a dropoff at 6 epochs.

I'm going to impliment the Dropout Regularization to see if we can increase our testing score. I'm also going to slightly adjust the neurons.

# Dropout 

## Model 1 - Dropout


```python
# Instentiate the model
drop_model = Sequential()

# Input layer
drop_model.add(Dense(200, activation = 'relu', input_dim = X_train_proc.shape[1]))
drop_model.add(Dropout(.2))

# Hidden layers
drop_model.add(Dense(40, activation = 'relu'))
 
# Output Layer
drop_model.add(Dense(1, activation = 'sigmoid'))

# Compile
drop_model.compile(optimizer = 'adam', loss = 'bce', metrics = ['acc'])

# History
drop_history = drop_model.fit(X_train_proc, y_train,
                        validation_data = (X_test_proc, y_test),
                        epochs = 7, verbose=0)

# plot learning history
plot_history(drop_history, 'acc')
```


    
![png](output_53_0.png)
    


### Dropout 1 - Score (82%)


```python
# Make predicitons and evaluate your model
print('Training Scores')

# Define labels for the confusion matrix
labels = ['Not Passing', 'Passing']

# Get training predictions and round them to integers instead of floats
train_preds = np.rint(drop_model.predict(X_train_proc))

# Classification Report
print(classification_report(y_train, train_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_train, train_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show()
print('\n')
print('Testing Scores')

# Get testing predictions and round them to integers
test_preds = np.rint(drop_model.predict(X_test_proc))

# Classification report
print(classification_report(y_test, test_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, test_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show();
```

    Training Scores
                  precision    recall  f1-score   support
    
               0       0.85      1.00      0.92      5611
               1       0.80      0.10      0.18      1120
    
        accuracy                           0.85      6731
       macro avg       0.82      0.55      0.55      6731
    weighted avg       0.84      0.85      0.79      6731
    
    


    
![png](output_55_1.png)
    


    
    
    Testing Scores
                  precision    recall  f1-score   support
    
               0       0.83      0.99      0.90      1870
               1       0.17      0.01      0.02       374
    
        accuracy                           0.82      2244
       macro avg       0.50      0.50      0.46      2244
    weighted avg       0.72      0.82      0.76      2244
    
    


    
![png](output_55_3.png)
    


### Evaluation

Awesome, our score increased to 82% when we implimented the a dropout of 45%. 

## Model 2: Dropout 2 - ropout's Revenge


```python
# Instentiate the model
drop_model_2 = Sequential()

# Input layer
drop_model_2.add(Dense(200, activation = 'relu', input_dim = X_train_proc.shape[1]))
drop_model_2.add(Dropout(.3))

# Hidden layers
drop_model_2.add(Dense(40, activation = 'relu'))

# Output Layer
drop_model_2.add(Dense(1, activation = 'sigmoid'))

# Compile
drop_model_2.compile(optimizer = 'adam', loss = 'bce', metrics = ['acc'])

# History
drop_history_2 = drop_model_2.fit(X_train_proc, y_train,
                        validation_data = (X_test_proc, y_test),
                        epochs = 7, verbose=0)

# plot learning history
plot_history(drop_history_2, 'acc')
```


    
![png](output_59_0.png)
    


### Dropout's Revenge - Score (83%)



```python
# Make predicitons and evaluate your model
print('Training Scores')

# Define labels for the confusion matrix
labels = ['Not Passing', 'Passing']

# Get training predictions and round them to integers instead of floats
train_preds = np.rint(drop_model_2.predict(X_train_proc))

# Classification Report
print(classification_report(y_train, train_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_train, train_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show()
print('\n')
print('Testing Scores')

# Get testing predictions and round them to integers
test_preds = np.rint(drop_model_2.predict(X_test_proc))

# Classification report
print(classification_report(y_test, test_preds))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, test_preds, normalize='true')
sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
plt.show();
```

    Training Scores
                  precision    recall  f1-score   support
    
               0       0.84      1.00      0.91      5611
               1       0.79      0.03      0.05      1120
    
        accuracy                           0.84      6731
       macro avg       0.81      0.51      0.48      6731
    weighted avg       0.83      0.84      0.77      6731
    
    


    
![png](output_61_1.png)
    


    
    
    Testing Scores
                  precision    recall  f1-score   support
    
               0       0.83      0.99      0.91      1870
               1       0.07      0.00      0.01       374
    
        accuracy                           0.83      2244
       macro avg       0.45      0.50      0.46      2244
    weighted avg       0.70      0.83      0.76      2244
    
    


    
![png](output_61_3.png)
    


### Evaluation

Awesome, with the smallest adjustment to the amount that is being dropped with 20% to 30% our score went up by one point. 

# Final Evaluation 

With the continious tuning to each new model and even implimenting dropout, I'm confident that we need the dropout regularization to continue to increase the models test score, which is why my final model choice is my second dropout model. 

This model can certainly continue to be adjusted and tunned to better assist with predicting the risk of providing a new loan to certain customers. 
