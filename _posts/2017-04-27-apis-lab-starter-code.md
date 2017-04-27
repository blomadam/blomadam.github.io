
# APIs Lab
In this lab we will practice using APIs to retrieve and store data.


```python
# Imports at the top
import json
import urllib
import pandas as pd
import numpy as np
import requests
import json
import re
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
from bs4 import BeautifulSoup
```


```python
import re
```


```python
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Imputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
```

## Exercise 1: Get Data From Sheetsu

[Sheetsu](https://sheetsu.com/) is an online service that allows you to access any Google spreadsheet from an API. This can be a very handy way to share a dataset with colleagues as well as to create a mini centralized data storage, that is simpler to edit than a database.

A Google Spreadsheet with wine data can be found [here](https://docs.google.com/a/generalassemb.ly/spreadsheets/d/1JWRwDnwIMLgvPqNMdJLmAJgzvz0K3zAUc6jev3ci1c8/edit?usp=sharing).

You can access it through the Sheetsu API at this endpoint: https://sheetsu.com/apis/v1.0/cc9420722ae4. [Here](https://sheetsu.com/docs/beta) is Sheetsu's documentation.


Questions:

1. Use the requests library to access the document. Inspect the response text. What kind of data is it?
2. Check the status code of the response object. What code is it?
3. Use the appropriate libraries and read functions to read the response into a Pandas Dataframe
4. Once you've imported the data into a dataframe, check the value of the 5th line: what's the price?


```python
URL = "https://sheetsu.com/apis/v1.0/cc9420722ae4"
response = requests.get(URL)
response.status_code  ## GET operation completed successfully!
```




    200




```python
data = response.json()  ## The data is in JSON format
df = pd.DataFrame(data)
df.head(10)
```




<div>
<table border="1" class="dataframe">
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




```python
df.tail(10)
```




<div>
<table border="1" class="dataframe">
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
      <th>212</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>213</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>214</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>215</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>216</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>217</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>218</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



> Answers:
    1. A JSON string.
    2. 200
    3. Options inlucde: pd.read_json; json.loads + pd.Dataframe
    4. 5

### Exercise 2: Post Data to Sheetsu
Now that we've learned how to read data, it'd be great if we could also write data. For this we will need to use a _POST_ request.

1. Use the post command to add the following data to the spreadsheet:


```python
post_data = {
'Grape' : ''
, 'Name' : 'Adam'
, 'Color' : 'W'
, 'Country' : 'US'
, 'Region' : 'Columbia'
, 'Vinyard' : 'Tigaris'
, 'Score' : '10'
, 'Consumed In' : '2015'
, 'Vintage' : '2012'
, 'Price' : '7'
}
```

1. What status did you get? How can you check that you actually added the data correctly?
- In this exercise, your classmates are adding data to the same spreadsheet. What happens because of this? Is it a problem? How could you mitigate it?


```python
post_response = requests.post(URL,json=post_data)
post_response.status_code
```




    201




```python
response2 = requests.get(URL)
response2.status_code
```




    200




```python
df2 = pd.DataFrame(response2.json())
df2.tail()
```




<div>
<table border="1" class="dataframe">
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
      <th>218</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>222</th>
      <td>W</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>Adam</td>
      <td>7</td>
      <td>Columbia</td>
      <td>10</td>
      <td>2012</td>
      <td>Tigaris</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (222, 10)



## Exercise 3: Data munging

Get back to the dataframe you've created in the beginning. Let's do some data munging:

1. Search for missing data
    - Is there any missing data? How do you deal with it?
    - Is there any data you can just remove?
    - Are the data types appropriate?
- Summarize the data 
    - Try using describe, min, max, mean, var


```python
# df2.info()
```


```python
df2.describe()
```




<div>
<table border="1" class="dataframe">
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
      <th>count</th>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
      <td>223</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>18</td>
      <td>27</td>
      <td>20</td>
      <td>22</td>
      <td>9</td>
      <td>7</td>
      <td>32</td>
    </tr>
    <tr>
      <th>top</th>
      <td>R</td>
      <td>2015</td>
      <td>US</td>
      <td></td>
      <td>My wonderful wine</td>
      <td>200</td>
      <td>Sonoma</td>
      <td>10</td>
      <td>1973</td>
      <td></td>
    </tr>
    <tr>
      <th>freq</th>
      <td>204</td>
      <td>211</td>
      <td>204</td>
      <td>200</td>
      <td>173</td>
      <td>187</td>
      <td>188</td>
      <td>193</td>
      <td>188</td>
      <td>185</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.drop_duplicates(inplace=True)
```


```python
df2[["Consumed In","Vintage","Price","Score"]] = df2[["Consumed In","Vintage","Price","Score"]].applymap(lambda x: np.nan if x=="" else x)
```


```python
df2.head() 
```




<div>
<table border="1" class="dataframe">
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
      <td>NaN</td>
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
  </tbody>
</table>
</div>




```python
# df2.info()
```


```python
# for i in df2.columns:
#     print df2[i].value_counts()
```


```python
df2.drop([106,148],inplace=True)
```


```python
df2.dropna(subset=["Score"],inplace=True)
df2.Score = df2.Score.map(float)
```


```python
df2["Consumed In"] = df2["Consumed In"].map(int)
```


```python
df2.Price = df2.Price.map(float)
```


```python
df2["Vintage"] = df2["Vintage"].map(int)
```


```python
df2.ix[2,"Country"] = "US"
df2.ix[11,"Country"] = "Italy"
df2.ix[25,"Country"] = "US"
```


```python
df2.Region.replace("Nappa","Napa", inplace=True)
df2.Region.replace("Columbia","Washington", inplace=True)
```


```python
# df2.info()
```


```python
df2.reset_index(inplace=True,drop=True)
# df2.info()
```


```python
# pd.get_dummies(df2,dummy_na=True,drop_first=True).info()  # do this with the pipeline now!
```

## Exercise 4: Feature Extraction

We would like to use a regression tree to predict the score of a wine. In order to do that, we first need to select and engineer appropriate features.

- Set the target to be the Score column, drop the rows with no score
- Use pd.get_dummies to create dummy features for all the text columns
- Fill the nan values in the numerical columns, using an appropriate method
- Train a Decision tree regressor on the Score, using a train test split:
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.3, random_state=42)
- Plot the test values, the predicted values and the residuals
- Calculate R^2 score
- Discuss your findings



```python
X = df2[[x for x in df2.columns if x !="Score"]]
y = df2.Score
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Imputer
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
# from sklearn.base import TransformerMixin, BaseEstimator
# from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
# from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn import tree
```


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



```python
pd.DataFrame(grid_search.cv_results_)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>mean_score_time</th>
      <th>mean_test_score</th>
      <th>mean_train_score</th>
      <th>param_features__Color__dummify__analyzer</th>
      <th>param_scale__model</th>
      <th>param_tree__max_depth</th>
      <th>param_tree__min_samples_split</th>
      <th>params</th>
      <th>rank_test_score</th>
      <th>split0_test_score</th>
      <th>split0_train_score</th>
      <th>split1_test_score</th>
      <th>split1_train_score</th>
      <th>split2_test_score</th>
      <th>split2_train_score</th>
      <th>std_fit_time</th>
      <th>std_score_time</th>
      <th>std_test_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.023624</td>
      <td>0.009839</td>
      <td>0.816376</td>
      <td>0.982000</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>2</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>18</td>
      <td>0.613658</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.002937</td>
      <td>0.002521</td>
      <td>0.177288</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.021573</td>
      <td>0.008326</td>
      <td>0.963752</td>
      <td>0.982000</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>2</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>1</td>
      <td>0.953758</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.008521</td>
      <td>0.001250</td>
      <td>0.009070</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.015636</td>
      <td>0.008277</td>
      <td>0.816376</td>
      <td>0.982000</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>2</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>18</td>
      <td>0.613658</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.001644</td>
      <td>0.000945</td>
      <td>0.177288</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.017216</td>
      <td>0.008123</td>
      <td>0.816376</td>
      <td>0.982000</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>2</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>18</td>
      <td>0.613658</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.001330</td>
      <td>0.000784</td>
      <td>0.177288</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.015525</td>
      <td>0.007242</td>
      <td>0.835760</td>
      <td>0.989765</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>3</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>10</td>
      <td>0.677055</td>
      <td>0.995995</td>
      <td>0.969856</td>
      <td>0.989201</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.001022</td>
      <td>0.000109</td>
      <td>0.139155</td>
      <td>0.004872</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.017921</td>
      <td>0.007588</td>
      <td>0.835760</td>
      <td>0.988829</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>3</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>10</td>
      <td>0.677055</td>
      <td>0.995995</td>
      <td>0.969856</td>
      <td>0.986393</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.003023</td>
      <td>0.000468</td>
      <td>0.139155</td>
      <td>0.005153</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.014828</td>
      <td>0.007339</td>
      <td>0.954595</td>
      <td>0.988829</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>3</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>7</td>
      <td>0.951288</td>
      <td>0.995995</td>
      <td>0.969856</td>
      <td>0.986393</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.000275</td>
      <td>0.000096</td>
      <td>0.010570</td>
      <td>0.005153</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.014769</td>
      <td>0.007268</td>
      <td>0.808071</td>
      <td>0.987545</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>3</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>24</td>
      <td>0.613156</td>
      <td>0.992143</td>
      <td>0.969856</td>
      <td>0.986393</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.000312</td>
      <td>0.000110</td>
      <td>0.170751</td>
      <td>0.003383</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.015645</td>
      <td>0.008688</td>
      <td>0.834897</td>
      <td>0.993249</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>4</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>13</td>
      <td>0.680439</td>
      <td>1.000000</td>
      <td>0.966250</td>
      <td>0.991471</td>
      <td>0.938119</td>
      <td>0.988275</td>
      <td>0.001030</td>
      <td>0.001012</td>
      <td>0.135482</td>
      <td>0.004949</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.015759</td>
      <td>0.008394</td>
      <td>0.838367</td>
      <td>0.992313</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>4</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>9</td>
      <td>0.684972</td>
      <td>1.000000</td>
      <td>0.966250</td>
      <td>0.988663</td>
      <td>0.943764</td>
      <td>0.988275</td>
      <td>0.000909</td>
      <td>0.000898</td>
      <td>0.134405</td>
      <td>0.005438</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.015764</td>
      <td>0.007797</td>
      <td>0.833565</td>
      <td>0.991902</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>4</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>15</td>
      <td>0.673892</td>
      <td>0.998768</td>
      <td>0.966250</td>
      <td>0.988663</td>
      <td>0.943764</td>
      <td>0.988275</td>
      <td>0.001156</td>
      <td>0.000272</td>
      <td>0.139886</td>
      <td>0.004857</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.014441</td>
      <td>0.007553</td>
      <td>0.807644</td>
      <td>0.990618</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>4</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>26</td>
      <td>0.617547</td>
      <td>0.994916</td>
      <td>0.966250</td>
      <td>0.988663</td>
      <td>0.938119</td>
      <td>0.988275</td>
      <td>0.000284</td>
      <td>0.000422</td>
      <td>0.166570</td>
      <td>0.003043</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.015308</td>
      <td>0.007625</td>
      <td>0.791434</td>
      <td>1.000000</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>None</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>31</td>
      <td>0.580718</td>
      <td>1.000000</td>
      <td>0.963084</td>
      <td>1.000000</td>
      <td>0.940741</td>
      <td>1.000000</td>
      <td>0.000905</td>
      <td>0.000401</td>
      <td>0.184457</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.014890</td>
      <td>0.007992</td>
      <td>0.955670</td>
      <td>0.998596</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>None</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>4</td>
      <td>0.968271</td>
      <td>1.000000</td>
      <td>0.963084</td>
      <td>0.997192</td>
      <td>0.926852</td>
      <td>0.998597</td>
      <td>0.000366</td>
      <td>0.000850</td>
      <td>0.017514</td>
      <td>0.001146</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.017532</td>
      <td>0.007906</td>
      <td>0.834227</td>
      <td>0.996860</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>None</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>14</td>
      <td>0.681447</td>
      <td>0.998768</td>
      <td>0.954471</td>
      <td>0.995320</td>
      <td>0.947222</td>
      <td>0.996493</td>
      <td>0.000724</td>
      <td>0.000512</td>
      <td>0.133631</td>
      <td>0.001431</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.015086</td>
      <td>0.007634</td>
      <td>0.955079</td>
      <td>0.995226</td>
      <td>char</td>
      <td>StandardScaler(copy=True, with_mean=True, with...</td>
      <td>None</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>5</td>
      <td>0.957647</td>
      <td>0.994916</td>
      <td>0.962264</td>
      <td>0.995320</td>
      <td>0.942824</td>
      <td>0.995440</td>
      <td>0.000466</td>
      <td>0.000335</td>
      <td>0.007642</td>
      <td>0.000224</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.014677</td>
      <td>0.007952</td>
      <td>0.816376</td>
      <td>0.982000</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>2</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>18</td>
      <td>0.613658</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.000253</td>
      <td>0.000986</td>
      <td>0.177288</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.015297</td>
      <td>0.007337</td>
      <td>0.963752</td>
      <td>0.982000</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>2</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>1</td>
      <td>0.953758</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.000383</td>
      <td>0.000108</td>
      <td>0.009070</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.017936</td>
      <td>0.008530</td>
      <td>0.816376</td>
      <td>0.982000</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>2</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>18</td>
      <td>0.613658</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.000094</td>
      <td>0.000835</td>
      <td>0.177288</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.015031</td>
      <td>0.007362</td>
      <td>0.816376</td>
      <td>0.982000</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>2</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>18</td>
      <td>0.613658</td>
      <td>0.987111</td>
      <td>0.968358</td>
      <td>0.981949</td>
      <td>0.974812</td>
      <td>0.976939</td>
      <td>0.000596</td>
      <td>0.000183</td>
      <td>0.177288</td>
      <td>0.004153</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.015325</td>
      <td>0.007354</td>
      <td>0.958414</td>
      <td>0.989765</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>3</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>3</td>
      <td>0.960102</td>
      <td>0.995995</td>
      <td>0.969856</td>
      <td>0.989201</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.000798</td>
      <td>0.000102</td>
      <td>0.010273</td>
      <td>0.004872</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.015453</td>
      <td>0.007733</td>
      <td>0.832487</td>
      <td>0.988829</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>3</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>16</td>
      <td>0.669501</td>
      <td>0.995995</td>
      <td>0.969856</td>
      <td>0.986393</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.000263</td>
      <td>0.000413</td>
      <td>0.142889</td>
      <td>0.005153</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.025059</td>
      <td>0.014044</td>
      <td>0.835760</td>
      <td>0.988829</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>3</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>10</td>
      <td>0.677055</td>
      <td>0.995995</td>
      <td>0.969856</td>
      <td>0.986393</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.006202</td>
      <td>0.003016</td>
      <td>0.139155</td>
      <td>0.005153</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.020851</td>
      <td>0.011469</td>
      <td>0.808071</td>
      <td>0.987545</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>3</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>24</td>
      <td>0.613156</td>
      <td>0.992143</td>
      <td>0.969856</td>
      <td>0.986393</td>
      <td>0.942798</td>
      <td>0.984100</td>
      <td>0.003194</td>
      <td>0.003551</td>
      <td>0.170751</td>
      <td>0.003383</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.016321</td>
      <td>0.008820</td>
      <td>0.788656</td>
      <td>0.993249</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>4</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>32</td>
      <td>0.580718</td>
      <td>1.000000</td>
      <td>0.956156</td>
      <td>0.991471</td>
      <td>0.938119</td>
      <td>0.988275</td>
      <td>0.001330</td>
      <td>0.000485</td>
      <td>0.181963</td>
      <td>0.004949</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.021296</td>
      <td>0.007924</td>
      <td>0.793190</td>
      <td>0.992313</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>4</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>29</td>
      <td>0.580718</td>
      <td>1.000000</td>
      <td>0.966250</td>
      <td>0.988663</td>
      <td>0.943764</td>
      <td>0.988275</td>
      <td>0.006233</td>
      <td>0.000210</td>
      <td>0.185993</td>
      <td>0.005438</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.022843</td>
      <td>0.009851</td>
      <td>0.952498</td>
      <td>0.991902</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>4</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>8</td>
      <td>0.951147</td>
      <td>0.998768</td>
      <td>0.967231</td>
      <td>0.988663</td>
      <td>0.938119</td>
      <td>0.988275</td>
      <td>0.006272</td>
      <td>0.003061</td>
      <td>0.011002</td>
      <td>0.004857</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.016505</td>
      <td>0.008454</td>
      <td>0.955020</td>
      <td>0.990618</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>4</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>6</td>
      <td>0.957647</td>
      <td>0.994916</td>
      <td>0.966250</td>
      <td>0.988663</td>
      <td>0.938119</td>
      <td>0.988275</td>
      <td>0.001183</td>
      <td>0.000372</td>
      <td>0.010817</td>
      <td>0.003043</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.016261</td>
      <td>0.008693</td>
      <td>0.794386</td>
      <td>1.000000</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>None</td>
      <td>2</td>
      <td>{u'tree__min_samples_split': 2, u'tree__max_de...</td>
      <td>28</td>
      <td>0.585251</td>
      <td>1.000000</td>
      <td>0.963084</td>
      <td>1.000000</td>
      <td>0.944444</td>
      <td>1.000000</td>
      <td>0.000888</td>
      <td>0.000329</td>
      <td>0.183017</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.019564</td>
      <td>0.008963</td>
      <td>0.792291</td>
      <td>0.998362</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>None</td>
      <td>3</td>
      <td>{u'tree__min_samples_split': 3, u'tree__max_de...</td>
      <td>30</td>
      <td>0.585251</td>
      <td>1.000000</td>
      <td>0.959393</td>
      <td>0.996490</td>
      <td>0.940741</td>
      <td>0.998597</td>
      <td>0.001781</td>
      <td>0.000544</td>
      <td>0.181187</td>
      <td>0.001442</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.017212</td>
      <td>0.008629</td>
      <td>0.831938</td>
      <td>0.996860</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>None</td>
      <td>4</td>
      <td>{u'tree__min_samples_split': 4, u'tree__max_de...</td>
      <td>17</td>
      <td>0.673892</td>
      <td>0.998768</td>
      <td>0.957752</td>
      <td>0.995320</td>
      <td>0.947222</td>
      <td>0.996493</td>
      <td>0.000634</td>
      <td>0.000281</td>
      <td>0.138264</td>
      <td>0.001431</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.015853</td>
      <td>0.008714</td>
      <td>0.805862</td>
      <td>0.995226</td>
      <td>char</td>
      <td>MinMaxScaler(copy=True, feature_range=(0, 1))</td>
      <td>None</td>
      <td>5</td>
      <td>{u'tree__min_samples_split': 5, u'tree__max_de...</td>
      <td>27</td>
      <td>0.613014</td>
      <td>0.994916</td>
      <td>0.962674</td>
      <td>0.995320</td>
      <td>0.942824</td>
      <td>0.995440</td>
      <td>0.000660</td>
      <td>0.000675</td>
      <td>0.168805</td>
      <td>0.000224</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(zip(grid_search.cv_results_['mean_test_score'],\
                 grid_search.cv_results_['std_test_score']\
                )).sort_values(0,ascending=False).head(10)

```




<div>
<table border="1" class="dataframe">
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
grid_search.best_estimator_
```




    Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('Color', Pipeline(steps=[('text', SampleExtractor(vars=['Color'])), ('dummify', CountVectorizer(analyzer='char', binary=True, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            ...       min_weight_fraction_leaf=0.0, presort=False, random_state=None,
               splitter='best'))])




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


![png](../images/apis-lab-starter-code_files/apis-lab-starter-code_46_0.png)



```python
plt.scatter(y_test,y_test.values-cv_pred.values,color='r')
plt.plot(y_test,y_test-y_test,color='k')
plt.xlabel("True value")
plt.ylabel("Residual")
plt.show()
```


![png](../images/apis-lab-starter-code_files/apis-lab-starter-code_47_0.png)


# Summary
I was able to run a grid search on the data, and found that two models fit the training data equally well.  When applied as a predictor on the test set, I get an R-squared value of 46.4%.  Looking at the plots, I seem to be over fit and predicting a wild outlier at the  right side.  There is also a linear pattern to my residuals, but the small sample size makes it hard to predict if that is by chance.

-------


---------

## Exercise 5: IMDB Movies

Sometimes an API doesn't provide all the information we would like to get and we need to be creative.
Here we will use a combination of scraping and API calls to investigate the ratings and gross earnings of famous movies.

## 5.a Get top movies

The Internet Movie Database contains data about movies. Unfortunately it does not have a public API.

The page http://www.imdb.com/chart/top contains the list of the top 250 movies of all times. Retrieve the page using the requests library and then parse the html to obtain a list of the `movie_ids` for these movies. You can parse it with regular expression or using a library like `BeautifulSoup`.

**Hint:** movie_ids look like this: `tt2582802`


```python
URL = "http://www.imdb.com/chart/top"
r = requests.get(URL)
print r.status_code
```

    200



```python
page_source = BeautifulSoup(r.content,"lxml")
```


```python
titles = page_source.findAll("td",class_="titleColumn")
```


```python
id_urls = []
for i in titles:
    id_urls.append(i.find("a")["href"])
```


```python
movie_ids = []
for i in id_urls:
    movie_ids.append(i[7:16]) # just noted the IDs are near the beginning of the url
```


```python
movie_ids[:5]  # check we have the IDs
```




    ['tt0111161', 'tt0068646', 'tt0071562', 'tt0468569', 'tt0050083']




```python
len(movie_ids)  #check we got them all
```




    250




```python

```

## 5.b Get top movies data

Although the Internet Movie Database does not have a public API, an open API exists at http://www.omdbapi.com.

Use this API to retrieve information about each of the 250 movies you have extracted in the previous step.
- Check the documentation of omdbapi.com to learn how to request movie data by id
- Define a function that returns a python object with all the information for a given id
- Iterate on all the IDs and store the results in a list of such objects
- Create a Pandas Dataframe from the list


```python
url_template = "http://www.omdbapi.com/?i={}"
data = []
for i in movie_ids:
    res = requests.get(url_template.format(i))
    data.append(res.json())
```


```python
df = pd.DataFrame(data)
```


```python
ratings = []
for item in df["Ratings"]:
    row = {}
    for dictionary in item:
        row[dictionary['Source']] = dictionary['Value']
    ratings.append(row)
print pd.DataFrame(ratings).head()
```

      Internet Movie Database Metacritic Rotten Tomatoes
    0                  9.3/10     80/100             91%
    1                  9.2/10    100/100             99%
    2                  9.0/10     80/100             97%
    3                  9.0/10     82/100             94%
    4                  8.9/10        NaN            100%



```python
review_df = pd.DataFrame(ratings)
```


```python
df2 = df.merge(review_df,left_index=True,right_index=True)
```


```python
df2.drop("Ratings",axis=1,inplace=True)
```


```python
df2 = df2.applymap(lambda x: x.encode('ascii',errors='ignore') if x==x else np.nan)
```


```python
df2.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actors</th>
      <th>Awards</th>
      <th>BoxOffice</th>
      <th>Country</th>
      <th>DVD</th>
      <th>Director</th>
      <th>Genre</th>
      <th>Language</th>
      <th>Metascore</th>
      <th>Plot</th>
      <th>...</th>
      <th>Type</th>
      <th>Website</th>
      <th>Writer</th>
      <th>Year</th>
      <th>imdbID</th>
      <th>imdbRating</th>
      <th>imdbVotes</th>
      <th>Internet Movie Database</th>
      <th>Metacritic</th>
      <th>Rotten Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tim Robbins, Morgan Freeman, Bob Gunton, Willi...</td>
      <td>Nominated for 7 Oscars. Another 19 wins &amp; 30 n...</td>
      <td>N/A</td>
      <td>USA</td>
      <td>27 Jan 1998</td>
      <td>Frank Darabont</td>
      <td>Crime, Drama</td>
      <td>English</td>
      <td>80</td>
      <td>Two imprisoned men bond over a number of years...</td>
      <td>...</td>
      <td>movie</td>
      <td>N/A</td>
      <td>Stephen King (short story "Rita Hayworth and S...</td>
      <td>1994</td>
      <td>tt0111161</td>
      <td>9.3</td>
      <td>1,786,262</td>
      <td>9.3/10</td>
      <td>80/100</td>
      <td>91%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marlon Brando, Al Pacino, James Caan, Richard ...</td>
      <td>Won 3 Oscars. Another 23 wins &amp; 27 nominations.</td>
      <td>N/A</td>
      <td>USA</td>
      <td>09 Oct 2001</td>
      <td>Francis Ford Coppola</td>
      <td>Crime, Drama</td>
      <td>English, Italian, Latin</td>
      <td>100</td>
      <td>The aging patriarch of an organized crime dyna...</td>
      <td>...</td>
      <td>movie</td>
      <td>http://www.thegodfather.com</td>
      <td>Mario Puzo (screenplay), Francis Ford Coppola ...</td>
      <td>1972</td>
      <td>tt0068646</td>
      <td>9.2</td>
      <td>1,227,935</td>
      <td>9.2/10</td>
      <td>100/100</td>
      <td>99%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Al Pacino, Robert Duvall, Diane Keaton, Robert...</td>
      <td>Won 6 Oscars. Another 10 wins &amp; 20 nominations.</td>
      <td>N/A</td>
      <td>USA</td>
      <td>24 May 2005</td>
      <td>Francis Ford Coppola</td>
      <td>Crime, Drama</td>
      <td>English, Italian, Spanish, Latin, Sicilian</td>
      <td>80</td>
      <td>The early life and career of Vito Corleone in ...</td>
      <td>...</td>
      <td>movie</td>
      <td>http://www.thegodfather.com/</td>
      <td>Francis Ford Coppola (screenplay), Mario Puzo ...</td>
      <td>1974</td>
      <td>tt0071562</td>
      <td>9.0</td>
      <td>845,231</td>
      <td>9.0/10</td>
      <td>80/100</td>
      <td>97%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Christian Bale, Heath Ledger, Aaron Eckhart, M...</td>
      <td>Won 2 Oscars. Another 147 wins &amp; 144 nominations.</td>
      <td>$533,316,061.00</td>
      <td>USA, UK</td>
      <td>09 Dec 2008</td>
      <td>Christopher Nolan</td>
      <td>Action, Crime, Drama</td>
      <td>English, Mandarin</td>
      <td>82</td>
      <td>When the menace known as the Joker wreaks havo...</td>
      <td>...</td>
      <td>movie</td>
      <td>http://thedarkknight.warnerbros.com/</td>
      <td>Jonathan Nolan (screenplay), Christopher Nolan...</td>
      <td>2008</td>
      <td>tt0468569</td>
      <td>9.0</td>
      <td>1,780,245</td>
      <td>9.0/10</td>
      <td>82/100</td>
      <td>94%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Martin Balsam, John Fiedler, Lee J. Cobb, E.G....</td>
      <td>Nominated for 3 Oscars. Another 16 wins &amp; 8 no...</td>
      <td>N/A</td>
      <td>USA</td>
      <td>06 Mar 2001</td>
      <td>Sidney Lumet</td>
      <td>Crime, Drama</td>
      <td>English</td>
      <td>N/A</td>
      <td>A jury holdout attempts to prevent a miscarria...</td>
      <td>...</td>
      <td>movie</td>
      <td>http://www.criterion.com/films/27871-12-angry-men</td>
      <td>Reginald Rose (story), Reginald Rose (screenplay)</td>
      <td>1957</td>
      <td>tt0050083</td>
      <td>8.9</td>
      <td>485,954</td>
      <td>8.9/10</td>
      <td>NaN</td>
      <td>100%</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
df2.to_csv("movies.csv",index=False)  # looks like imdb and metacritic ratings are duplicate columns
```

## 5.c Get gross data

The OMDB API is great, but it does not provide information about Gross Revenue of the movie. We'll revert back to scraping for this.

- Write a function that retrieves the gross revenue from the entry page at imdb.com
- The function should handle the exception of when the page doesn't report gross revenue
- Retrieve the gross revenue for each movie and store it in a separate dataframe


```python
url_template = "http://www.imdb.com/title/{}"
r = requests.get(url_template.format(movie_ids[0]))  # get the first movie page for testing
r.status_code
```




    200




```python
import re
```


```python
mo = re.search(r'Gross\:.*\$(.*)', r.content) 
# search through the webpage for "Gross:" and  capture the number after the following dollar sign ($)
```


```python
int(mo.group(1).strip().replace(",",""))  # try to convert the gross earnings to a number
```




    28341469




```python
gross = []   # now implement a loop to collect this data from each webpage (add None on failure)
for i in movie_ids:
    r = requests.get(url_template.format(i))
    try:
        gross.append(int(re.search(r'Gross\:.*\$(.*)', r.content).group(1).strip().replace(",","")))
    except:
        gross.append(None)
```


```python
print gross[:5] # check the numbers make sense
print len(gross)
```

    [28341469, 134821952, 57300000, 533316061, None]
    250



```python
gross_db = pd.DataFrame(zip(movie_ids,gross),columns=["imdbID",'Gross']) # save to dataframe
gross_db.to_csv("gross.csv",index=False)
```

## 5.d Data munging

- Now that you have movie information and gross revenue information, let's clean the two datasets.
- Check if there are null values. Be careful they may appear to be valid strings.
- Convert the columns to the appropriate formats. In particular handle:
    - Released
    - Runtime
    - year
    - imdbRating
    - imdbVotes
- Merge the data from the two datasets into a single one


```python
gross_db.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 250 entries, 0 to 249
    Data columns (total 2 columns):
    imdbID    250 non-null object
    Gross     184 non-null float64
    dtypes: float64(1), object(1)
    memory usage: 4.0+ KB



```python
df2.Released.replace("N/A",None,inplace=True)
```


```python
df2.Released = df2.Released.map(lambda x: pd.to_datetime(x, format="%d %b %Y") if x==x else x)
```


```python
df2.Runtime = df2.Runtime.map(lambda x: int(x.strip().split()[0]))
```


```python
df2.Year = df2.Year.map(lambda x: int(x))
```


```python
df2.imdbRating = df2.imdbRating.map(lambda x: float(x))
```


```python
df2.imdbVotes = df2.imdbVotes.map(lambda x: int(x.strip().replace(",","")))
```


```python
movie_df = df2.merge(gross_db, on="imdbID")
movie_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actors</th>
      <th>Awards</th>
      <th>BoxOffice</th>
      <th>Country</th>
      <th>DVD</th>
      <th>Director</th>
      <th>Genre</th>
      <th>Language</th>
      <th>Metascore</th>
      <th>Plot</th>
      <th>...</th>
      <th>Website</th>
      <th>Writer</th>
      <th>Year</th>
      <th>imdbID</th>
      <th>imdbRating</th>
      <th>imdbVotes</th>
      <th>Internet Movie Database</th>
      <th>Metacritic</th>
      <th>Rotten Tomatoes</th>
      <th>Gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tim Robbins, Morgan Freeman, Bob Gunton, Willi...</td>
      <td>Nominated for 7 Oscars. Another 19 wins &amp; 30 n...</td>
      <td>N/A</td>
      <td>USA</td>
      <td>27 Jan 1998</td>
      <td>Frank Darabont</td>
      <td>Crime, Drama</td>
      <td>English</td>
      <td>80</td>
      <td>Two imprisoned men bond over a number of years...</td>
      <td>...</td>
      <td>N/A</td>
      <td>Stephen King (short story "Rita Hayworth and S...</td>
      <td>1994</td>
      <td>tt0111161</td>
      <td>9.3</td>
      <td>1786262</td>
      <td>9.3/10</td>
      <td>80/100</td>
      <td>91%</td>
      <td>28341469.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marlon Brando, Al Pacino, James Caan, Richard ...</td>
      <td>Won 3 Oscars. Another 23 wins &amp; 27 nominations.</td>
      <td>N/A</td>
      <td>USA</td>
      <td>09 Oct 2001</td>
      <td>Francis Ford Coppola</td>
      <td>Crime, Drama</td>
      <td>English, Italian, Latin</td>
      <td>100</td>
      <td>The aging patriarch of an organized crime dyna...</td>
      <td>...</td>
      <td>http://www.thegodfather.com</td>
      <td>Mario Puzo (screenplay), Francis Ford Coppola ...</td>
      <td>1972</td>
      <td>tt0068646</td>
      <td>9.2</td>
      <td>1227935</td>
      <td>9.2/10</td>
      <td>100/100</td>
      <td>99%</td>
      <td>134821952.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Al Pacino, Robert Duvall, Diane Keaton, Robert...</td>
      <td>Won 6 Oscars. Another 10 wins &amp; 20 nominations.</td>
      <td>N/A</td>
      <td>USA</td>
      <td>24 May 2005</td>
      <td>Francis Ford Coppola</td>
      <td>Crime, Drama</td>
      <td>English, Italian, Spanish, Latin, Sicilian</td>
      <td>80</td>
      <td>The early life and career of Vito Corleone in ...</td>
      <td>...</td>
      <td>http://www.thegodfather.com/</td>
      <td>Francis Ford Coppola (screenplay), Mario Puzo ...</td>
      <td>1974</td>
      <td>tt0071562</td>
      <td>9.0</td>
      <td>845231</td>
      <td>9.0/10</td>
      <td>80/100</td>
      <td>97%</td>
      <td>57300000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Christian Bale, Heath Ledger, Aaron Eckhart, M...</td>
      <td>Won 2 Oscars. Another 147 wins &amp; 144 nominations.</td>
      <td>$533,316,061.00</td>
      <td>USA, UK</td>
      <td>09 Dec 2008</td>
      <td>Christopher Nolan</td>
      <td>Action, Crime, Drama</td>
      <td>English, Mandarin</td>
      <td>82</td>
      <td>When the menace known as the Joker wreaks havo...</td>
      <td>...</td>
      <td>http://thedarkknight.warnerbros.com/</td>
      <td>Jonathan Nolan (screenplay), Christopher Nolan...</td>
      <td>2008</td>
      <td>tt0468569</td>
      <td>9.0</td>
      <td>1780245</td>
      <td>9.0/10</td>
      <td>82/100</td>
      <td>94%</td>
      <td>533316061.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Martin Balsam, John Fiedler, Lee J. Cobb, E.G....</td>
      <td>Nominated for 3 Oscars. Another 16 wins &amp; 8 no...</td>
      <td>N/A</td>
      <td>USA</td>
      <td>06 Mar 2001</td>
      <td>Sidney Lumet</td>
      <td>Crime, Drama</td>
      <td>English</td>
      <td>N/A</td>
      <td>A jury holdout attempts to prevent a miscarria...</td>
      <td>...</td>
      <td>http://www.criterion.com/films/27871-12-angry-men</td>
      <td>Reginald Rose (story), Reginald Rose (screenplay)</td>
      <td>1957</td>
      <td>tt0050083</td>
      <td>8.9</td>
      <td>485954</td>
      <td>8.9/10</td>
      <td>NaN</td>
      <td>100%</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
movie_df.Actors.head()
```




    0    Tim Robbins, Morgan Freeman, Bob Gunton, Willi...
    1    Marlon Brando, Al Pacino, James Caan, Richard ...
    2    Al Pacino, Robert Duvall, Diane Keaton, Robert...
    3    Christian Bale, Heath Ledger, Aaron Eckhart, M...
    4    Martin Balsam, John Fiedler, Lee J. Cobb, E.G....
    Name: Actors, dtype: object



## 5.d Text vectorization

There are several columns in the data that contain a comma separated list of items, for example the Genre column and the Actors column. Let's transform those to binary columns using the count vectorizer from scikit learn.

Append these columns to the merged dataframe.

**Hint:** In order to get the actors name right, you'll have to set the `token_pattern` parameter in `CountVectorizer` to u'(?u)\\w+\.?\\w?\.? \\w+'. Can you see why? How does this differ from the default?

## notes for `token_pattern`

`u'(?u)\b\w\w+\b'`       default = 2+ word chars between word boundaries
`u'(?u)\w+.?\w?.? \w+'`  Actors = (1 or more letters) + (0 or 1 char) + (0 or 1 letter) + (0 or 1 char) + space + (1 or more letters)

I want 1 or more letters + 0 or 1 char + 1 or more letters:
    `'(?u)\w+.?\w+'`

Dictionary:
- `(?u)` re.U (Unicode dependent) i.e. U from re module
- `\w` any of [a-zA-Z0-9_]
- `\b` word boundary (like ^ or $ for lines)
- `.` any character
- `?` 0 or 1 match of previous char
- `+` one or more repeats of the previous character



```python
v = CountVectorizer(
            binary= True,
            token_pattern= u'(?u)\w+.?\w?.? \w+'
    )
data = v.fit_transform(movie_df.Actors).todense()
names = v.get_feature_names()

temp_df = pd.DataFrame(data,columns=names)

for i in temp_df.columns:
    movie_df.insert(len(movie_df.columns)-1,"actor_"+i,temp_df[i].values)

v = CountVectorizer(
            binary= True,
            token_pattern= '(?u)\w+.?\w+'
    )
data = v.fit_transform(movie_df.Genre).todense()
names = v.get_feature_names()

temp_df = pd.DataFrame(data,columns=names)

for i in temp_df.columns:
    movie_df.insert(len(movie_df.columns)-1,"genre_"+i,temp_df[i].values)

```


```python
movie_df.shape
```




    (250, 880)



## Bonus:

- What are the top 10 grossing movies?
- Who are the 10 actors that appear in the most movies?
- What's the average grossing of the movies in which each of these actors appear?
- What genre is the oldest movie?



```python
# What are the top 10 grossing movies?
movie_df[["Title","Gross"]].sort_values("Gross",ascending=False).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211</th>
      <td>Star Wars: The Force Awakens</td>
      <td>936627416.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight</td>
      <td>533316061.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Star Wars: Episode IV - A New Hope</td>
      <td>460935665.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>The Dark Knight Rises</td>
      <td>448130642.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>The Lion King</td>
      <td>422783777.0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Toy Story 3</td>
      <td>414984497.0</td>
    </tr>
    <tr>
      <th>218</th>
      <td>Harry Potter and the Deathly Hallows: Part 2</td>
      <td>380955619.0</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Finding Nemo</td>
      <td>380838870.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Lord of the Rings: The Return of the King</td>
      <td>377019252.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Jurassic Park</td>
      <td>356784000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Who are the 10 actors that appear in the most movies?
movie_df.iloc[:,27:-22].apply(np.sum).sort_values(ascending=False).head(14)  # had several tied in last spot
```




    actor_robert de niro       7
    actor_harrison ford        7
    actor_leonardo dicaprio    6
    actor_tom hanks            6
    actor_aamir khan           6
    actor_clint eastwood       5
    actor_joe pesci            4
    actor_morgan freeman       4
    actor_christian bale       4
    actor_al pacino            4
    actor_james stewart        4
    actor_charles chaplin      4
    actor_carrie fisher        4
    actor_mark hamill          4
    dtype: int64




```python
# What's the average grossing of the movies in which each of these actors appear?
print("{:25s}{}").format("Actor", "AVG Movie Gross")
print "-"*40
for actor in movie_df.iloc[:,27:-22].apply(np.sum).sort_values(ascending=False).head(14).index:
    print("{:25s} $ {:>12,.0f}").format(actor, movie_df[movie_df[actor]==1].Gross.mean())
```

    Actor                    AVG Movie Gross
    ----------------------------------------
    actor_robert de niro      $   36,559,324
    actor_harrison ford       $  351,913,357
    actor_leonardo dicaprio   $  166,169,549
    actor_tom hanks           $  242,304,669
    actor_aamir khan          $    6,329,966
    actor_clint eastwood      $   88,941,497
    actor_joe pesci           $   23,654,986
    actor_morgan freeman      $   82,511,760
    actor_christian bale      $  309,968,305
    actor_al pacino           $   76,064,488
    actor_james stewart       $   13,850,000
    actor_charles chaplin     $    1,331,622
    actor_carrie fisher       $  499,211,810
    actor_mark hamill         $  499,211,810



```python
# What genre is the oldest movie?
movie_df[["Title","Released","Gross"]].sort_values("Released",ascending=True).head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Released</th>
      <th>Gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>The Kid</td>
      <td>1921-02-06</td>
      <td>2500000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
