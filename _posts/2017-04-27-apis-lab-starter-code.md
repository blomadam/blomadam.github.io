---
layout: post
title: Grid searching data pipelines for wine quality
date: 2017-05-10
published: true
categories: projects
image: /images/apis-lab-starter-code_files/apis-lab-starter-code_46_0.png
---

I collected a small set of crowd sourced wine data from the web to show how pipelines and grid search can be used together to run data analysis while being careful not to leak information from the test set into the training data.

##Load the data

```python
data = response.json()  ## The data is in JSON format
df = pd.DataFrame(data)
df.head(10)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Color</th>
      <th>Consumed In</th>
      <th>Country</th>
      <th>Grape</th>
      <th>Name</th>
      <th>Price</th>
      <th>Region</th>
      <th>Score</th>
      <th>Vintage</th>
      <th>Vinyard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>W</td>
      <td>2015</td>
      <td>Portugal</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Portugal</td>
      <td>4</td>
      <td>2013</td>
      <td>Vinho Verde</td>
    </tr>
    <tr>
      <th>1</th>
      <td>W</td>
      <td>2015</td>
      <td>France</td>
      <td></td>
      <td></td>
      <td>17.8</td>
      <td>France</td>
      <td>3</td>
      <td>2013</td>
      <td>Peyruchet</td>
    </tr>
    <tr>
      <th>2</th>
      <td>W</td>
      <td>2015</td>
      <td>Oregon</td>
      <td></td>
      <td></td>
      <td>20</td>
      <td>Oregon</td>
      <td>3</td>
      <td>2013</td>
      <td>Abacela</td>
    </tr>
    <tr>
      <th>3</th>
      <td>W</td>
      <td>2015</td>
      <td>Spain</td>
      <td>chardonay</td>
      <td></td>
      <td>7</td>
      <td>Spain</td>
      <td>2.5</td>
      <td>2012</td>
      <td>Ochoa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td>chiraz, cab</td>
      <td>Spice Trader</td>
      <td>6</td>
      <td></td>
      <td>3</td>
      <td>2012</td>
      <td>Heartland</td>
    </tr>
    <tr>
      <th>5</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td>cab</td>
      <td></td>
      <td>13</td>
      <td>California</td>
      <td>3.5</td>
      <td>2012</td>
      <td>Crow Canyon</td>
    </tr>
    <tr>
      <th>6</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>#14</td>
      <td>21</td>
      <td>Oregon</td>
      <td>2.5</td>
      <td>2013</td>
      <td>Abacela</td>
    </tr>
    <tr>
      <th>7</th>
      <td>R</td>
      <td>2015</td>
      <td>France</td>
      <td>merlot, cab</td>
      <td></td>
      <td>12</td>
      <td>Bordeaux</td>
      <td>3.5</td>
      <td>2012</td>
      <td>David Beaulieu</td>
    </tr>
    <tr>
      <th>8</th>
      <td>R</td>
      <td>2015</td>
      <td>France</td>
      <td>merlot, cab</td>
      <td></td>
      <td>11.99</td>
      <td>Medoc</td>
      <td>3.5</td>
      <td>2011</td>
      <td>Chantemerle</td>
    </tr>
    <tr>
      <th>9</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td>merlot</td>
      <td></td>
      <td>13</td>
      <td>Washington</td>
      <td>4</td>
      <td>2011</td>
      <td>Hyatt</td>
    </tr>
  </tbody>
</table>
</div>

## Split the data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Set up classes for custom pipeline

```python
class ModelTransformer(BaseEstimator,TransformerMixin):

    def __init__(self, model=None):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return self.model.transform(X)
    
class SampleExtractor(BaseEstimator, TransformerMixin):
    """Takes in varaible names as a **list**"""

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column names to extract

    def transform(self, X, y=None):
        if len(self.vars) > 1:
            return pd.DataFrame(X[self.vars]) # where the actual feature extraction happens
        else:
            return pd.Series(X[self.vars[0]])

    def fit(self, X, y=None):
        return self  # generally does nothing
    
    
class DenseTransformer(BaseEstimator,TransformerMixin):

    def transform(self, X, y=None, **fit_params):
#         print X.todense()
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
```


## Run data with KFolds cross validation

```python
kf_shuffle = StratifiedKFold(n_splits=3,shuffle=True,random_state=777)

binary = True
feats = 5

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('Color', Pipeline([
                      ('text',SampleExtractor(['Color'])),
                      ('dummify', CountVectorizer(binary=binary, max_features=feats)),
                      ('densify', DenseTransformer()),
                     ])),
        ('Country', Pipeline([
                      ('text',SampleExtractor(['Country'])),
                      ('dummify', CountVectorizer(binary=binary, max_features=feats)),
                      ('densify', DenseTransformer()),
                     ])),
        ('Grape', Pipeline([
                      ('text',SampleExtractor(['Grape'])),
                      ('dummify', CountVectorizer(binary=binary, max_features=feats)),
                      ('densify', DenseTransformer()),
                     ])),
        ('Name', Pipeline([
                      ('text',SampleExtractor(['Name'])),
                      ('dummify', CountVectorizer(binary=binary, max_features=feats)),
                      ('densify', DenseTransformer()),
                     ])),
        ('Region', Pipeline([
                      ('text',SampleExtractor(['Region'])),
                      ('dummify', CountVectorizer(binary=binary, max_features=feats)),
                      ('densify', DenseTransformer()),
                     ])),
        ('Vinyard', Pipeline([
                      ('text',SampleExtractor(['Vinyard'])),
                      ('dummify', CountVectorizer(binary=binary, max_features=feats)),
                      ('densify', DenseTransformer()),
                     ])),
        ('cont_features', Pipeline([
                      ('continuous', SampleExtractor(['Consumed In', 'Price', 'Vintage'])),
                      ('impute',Imputer()),
                      ])),
        ])),
        ('scale', ModelTransformer()),
        ('tree', tree.DecisionTreeRegressor()),
])


parameters = {
    'features__Color__dummify__analyzer':['char'],
    'scale__model': (StandardScaler(),MinMaxScaler()),
    'tree__max_depth': (2,3,4,None),
    'tree__min_samples_split': (2,3,4,5),
}

grid_search = GridSearchCV(pipeline, parameters, verbose=False, cv=kf_shuffle)

```


## Execute the pipeline

```python
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)


grid_search.fit(X_train, y_train)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


cv_pred = pd.Series(grid_search.predict(X_test))
```

    Performing grid search...
    ('pipeline:', ['features', 'scale', 'tree'])
    parameters:
    {'tree__min_samples_split': (2, 3, 4, 5), 'tree__max_depth': (2, 3, 4, None), 'scale__model': (StandardScaler(copy=True, with_mean=True, with_std=True), MinMaxScaler(copy=True, feature_range=(0, 1))), 'features__Color__dummify__analyzer': ['char']}
    Best score: 0.964
    Best parameters set:
    	features__Color__dummify__analyzer: 'char'
    	scale__model: StandardScaler(copy=True, with_mean=True, with_std=True)
    	tree__max_depth: 2
    	tree__min_samples_split: 3



## Take a peek at the results


```python
pd.DataFrame(zip(grid_search.cv_results_['mean_test_score'],\
                 grid_search.cv_results_['std_test_score']\
                )).sort_values(0,ascending=False).head(10)

```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>0.963752</td>
      <td>0.009070</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.963752</td>
      <td>0.009070</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.958414</td>
      <td>0.010273</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.955670</td>
      <td>0.017514</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.955079</td>
      <td>0.007642</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.955020</td>
      <td>0.010817</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.954595</td>
      <td>0.010570</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.952498</td>
      <td>0.011002</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.838367</td>
      <td>0.134405</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.835760</td>
      <td>0.139155</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid_search.score(X_test,y_test) # prints the R2 for the best predictor
```

    0.46448936196721591




```python
plt.scatter(y_test,cv_pred,color='r')
plt.plot(y_test,y_test,color='k')
plt.xlabel("True value")
plt.ylabel("Predicted Value")
plt.show()
```


![png](/images/apis-lab-starter-code_files/apis-lab-starter-code_46_0.png)



```python
plt.scatter(y_test,y_test.values-cv_pred.values,color='r')
plt.plot(y_test,y_test-y_test,color='k')
plt.xlabel("True value")
plt.ylabel("Residual")
plt.show()
```


![png](/images/apis-lab-starter-code_files/apis-lab-starter-code_47_0.png)


## Results Summary
I was able to run a grid search on the data, and found that two models fit the training data equally well.  When applied as a predictor on the test set, I get an R-squared value of 46.4%.  Looking at the plots, I seem to be over fit and predicting a wild outlier at the  right side.  There is also a linear pattern to my residuals from one model, but the small sample size makes it hard to predict if that is by chance.

