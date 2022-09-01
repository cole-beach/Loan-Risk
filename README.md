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
print(df.info(),'\n')
print(f'The shape of the Dataset is: {df.shape}')

print(f'There are {df.duplicated().sum()} duplicated values','\n')
print(f'===========================================\n\n Missing Values:')
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()

df.drop_duplicates(inplace=True)
print(f'After dropping duplicated values we have {df.duplicated().sum()} shown in the dataset')

df.describe().T

plt.figure(figsize = (24, 12))

corr = df.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()
