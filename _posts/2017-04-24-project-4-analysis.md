---
layout: post
title: US Salary Summary Predictions
date: 2017-04-23
published: false
categories: projects
image: /images/project4/Size.png
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

%matplotlib inline
```


```python
df = pd.read_csv("salary_data.csv")
```


```python
df.head()
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>search_city</th>
      <th>location</th>
      <th>company</th>
      <th>jobtitle</th>
      <th>summary</th>
      <th>date_posted</th>
      <th>days_ago_posted</th>
      <th>reviews</th>
      <th>ratings</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Francisco, CA</td>
      <td>Palo Alto, CA</td>
      <td>SAP</td>
      <td>Data Scientist - Machine Learning</td>
      <td>Implement most recent algorithms and approache...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>454.0</td>
      <td>52.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>San Francisco, CA</td>
      <td>Mountain View, CA 94043</td>
      <td>First Tech Federal Credit Union</td>
      <td>Data Scientist</td>
      <td>Identify what data is available and relevant, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.0</td>
      <td>39.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>San Francisco, CA</td>
      <td>Oakland, CA 94612</td>
      <td>Opinion Dynamics</td>
      <td>Quantitative Data Analyst</td>
      <td>Managing, aggregating and cleaning data sets i...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>31.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>San Francisco, CA</td>
      <td>San Francisco, CA</td>
      <td>Radius</td>
      <td>Junior Data Scientist</td>
      <td>Develop new methodologies for data validation....</td>
      <td>4 days ago</td>
      <td>4.0</td>
      <td>23.0</td>
      <td>42.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>San Francisco, CA</td>
      <td>San Francisco, CA 94104 (Financial District area)</td>
      <td>McKinsey &amp; Company</td>
      <td>Junior Data Scientist, Marketing</td>
      <td>Experience linking multiple data platforms (so...</td>
      <td>3 days ago</td>
      <td>3.0</td>
      <td>227.0</td>
      <td>52.8</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Handle null values


```python
df['days_ago_posted'].replace(np.nan,35,regex=True,inplace=True)
df['date_posted'].replace(np.nan,35,regex=True,inplace=True)
df.dropna(subset=["location","company","jobtitle","summary"],inplace=True)
```


```python
df.reset_index(inplace=True)
```


```python
df.insert(1,'state',df.search_city.map(lambda x: x.split(",")[-1].strip()))
```


```python
df.describe()
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>days_ago_posted</th>
      <th>reviews</th>
      <th>ratings</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6698.000000</td>
      <td>6698.000000</td>
      <td>4848.000000</td>
      <td>4848.000000</td>
      <td>426.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3351.546133</td>
      <td>20.072111</td>
      <td>870.834571</td>
      <td>47.018812</td>
      <td>84479.957746</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1934.830372</td>
      <td>11.201123</td>
      <td>2355.386556</td>
      <td>5.880968</td>
      <td>48603.491111</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>15.000000</td>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1676.250000</td>
      <td>9.000000</td>
      <td>15.000000</td>
      <td>43.200000</td>
      <td>47476.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3351.500000</td>
      <td>25.000000</td>
      <td>87.500000</td>
      <td>44.400000</td>
      <td>70000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5026.750000</td>
      <td>30.000000</td>
      <td>457.000000</td>
      <td>52.200000</td>
      <td>115000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6702.000000</td>
      <td>35.000000</td>
      <td>24407.000000</td>
      <td>60.000000</td>
      <td>260000.000000</td>
    </tr>
  </tbody>
</table>
</div>



## I need to classify my search locations into a more smaller set of features


```python
def boxplot_sorted(df, by, column):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    meds = df2.median().sort_values()
    df2[meds.index].boxplot(rot=90,figsize=(14,8),fontsize=12)
    plt.show()
```


```python
df[df['salary'].notnull()].boxplot(column="salary", by="search_city",figsize=(14,8),rot=90,fontsize=16)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111861a10>




![png](/images/project-4-analysis_files/project-4-analysis_10_1.png)



```python
boxplot_sorted(df[df['salary'].notnull()], by="search_city", column="salary")
```


![png](/images/project-4-analysis_files/project-4-analysis_11_0.png)



```python
boxplot_sorted(df[df['salary'].notnull()], by="state", column="salary")
```


![png](/images/project-4-analysis_files/project-4-analysis_12_0.png)



```python
df[df['salary'].notnull()].pivot_table(index=["search_city"], values=["salary"], \
               aggfunc=[np.median,np.std,len]).sort_values(("median","salary"))
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>median</th>
      <th>std</th>
      <th>len</th>
    </tr>
    <tr>
      <th></th>
      <th>salary</th>
      <th>salary</th>
      <th>salary</th>
    </tr>
    <tr>
      <th>search_city</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sherman Oaks, CA</th>
      <td>30000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Overland Park, KS</th>
      <td>34340.0</td>
      <td>13529.932249</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Carlsbad, CA</th>
      <td>38840.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Phoenix, AZ</th>
      <td>40000.0</td>
      <td>32700.752833</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Raleigh, NC</th>
      <td>41300.0</td>
      <td>14507.797056</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Portland, OR</th>
      <td>45000.0</td>
      <td>24044.393091</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Pittsburgh, PA</th>
      <td>45000.0</td>
      <td>63292.419704</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>San Diego, CA</th>
      <td>46590.0</td>
      <td>29314.184849</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>St. Louis, MO</th>
      <td>47476.0</td>
      <td>19936.522125</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>Marlborough, MA</th>
      <td>48508.5</td>
      <td>9204.408971</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Salt Lake City, UT</th>
      <td>48540.0</td>
      <td>20195.757687</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Newton, MA</th>
      <td>50000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>St Petersburg Beach, FL</th>
      <td>52045.0</td>
      <td>34066.024252</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Deerfield Beach, FL</th>
      <td>52185.5</td>
      <td>18026.122852</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Port Washington, NY</th>
      <td>52600.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Seattle, WA</th>
      <td>53500.0</td>
      <td>46012.527003</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Berkeley, CA</th>
      <td>54900.0</td>
      <td>51286.625288</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Cupertino, CA</th>
      <td>55000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Redmond, WA</th>
      <td>55250.0</td>
      <td>28381.331893</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Culver City, CA</th>
      <td>57274.0</td>
      <td>60869.889206</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Santa Monica, CA</th>
      <td>60000.0</td>
      <td>28844.410204</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Houston, TX</th>
      <td>62500.0</td>
      <td>59956.853363</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Denver, CO</th>
      <td>63339.0</td>
      <td>38087.664153</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Austin, TX</th>
      <td>64800.0</td>
      <td>25463.318189</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>Dallas, TX</th>
      <td>65000.0</td>
      <td>33881.543265</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Renton, WA</th>
      <td>65132.0</td>
      <td>48937.446112</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Fort George G Meade, MD</th>
      <td>66228.0</td>
      <td>22547.906098</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Union, NJ</th>
      <td>70000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Pasadena, CA</th>
      <td>73286.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Los Angeles, CA</th>
      <td>73402.0</td>
      <td>44686.123208</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>Schaumburg, IL</th>
      <td>75000.0</td>
      <td>36339.919422</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Detroit, MI</th>
      <td>80000.0</td>
      <td>42426.406871</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>South Hackensack, NJ</th>
      <td>80000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Chevy Chase, MD</th>
      <td>80000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Atlanta, GA</th>
      <td>82852.5</td>
      <td>42502.676768</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Torrance, CA</th>
      <td>84000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Henderson, NV</th>
      <td>85000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Washington, DC</th>
      <td>87923.0</td>
      <td>50109.018605</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Downers Grove, IL</th>
      <td>90000.0</td>
      <td>23874.672773</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Irvine, CA</th>
      <td>90000.0</td>
      <td>42875.503673</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>New York, NY</th>
      <td>100000.0</td>
      <td>52793.061512</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>Mountain View, CA</th>
      <td>100000.0</td>
      <td>36055.512755</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Cambridge, MA</th>
      <td>100000.0</td>
      <td>48673.858954</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>San Jose, CA</th>
      <td>100000.0</td>
      <td>34880.749227</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>South Plainfield, NJ</th>
      <td>106250.0</td>
      <td>62975.866976</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Chicago, IL</th>
      <td>117500.0</td>
      <td>33486.981930</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>Jersey City, NJ</th>
      <td>117500.0</td>
      <td>37143.303851</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Palo Alto, CA</th>
      <td>120000.0</td>
      <td>63825.526464</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Philadelphia, PA</th>
      <td>130000.0</td>
      <td>60107.328248</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>San Francisco, CA</th>
      <td>140000.0</td>
      <td>66852.832775</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>McLean, VA</th>
      <td>140000.0</td>
      <td>49673.936828</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Redwood City, CA</th>
      <td>140000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Boston, MA</th>
      <td>140000.0</td>
      <td>19344.249792</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Manhattan, NY</th>
      <td>150000.0</td>
      <td>59450.622649</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>San Ramon, CA</th>
      <td>172500.0</td>
      <td>47675.115801</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Campbell, CA</th>
      <td>180000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Natick, MA</th>
      <td>200000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>San Mateo, CA</th>
      <td>260000.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Check in each state for dissimilar cities


```python
for i in df.state.unique():
    df[df['state']==i].boxplot(column="salary", by="search_city",figsize=(14,8),rot=90,fontsize=16)
    plt.ylim(0,275000)
    plt.show()
```


![png](/images/project-4-analysis_files/project-4-analysis_15_0.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_1.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_2.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_3.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_4.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_5.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_6.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_7.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_8.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_9.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_10.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_11.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_12.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_13.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_14.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_15.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_16.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_17.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_18.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_19.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_20.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_21.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_22.png)



![png](/images/project-4-analysis_files/project-4-analysis_15_23.png)



```python
df[df['salary'].notnull()].state.value_counts()
```




    CA    83
    TX    47
    IL    46
    NY    25
    MO    23
    MA    20
    AZ    20
    NC    20
    WA    19
    NJ    18
    PA    16
    MD    14
    DC    14
    GA    10
    OR     9
    FL     9
    UT     7
    CO     7
    KS     7
    MI     6
    VA     5
    NV     1
    Name: state, dtype: int64



### Check for clusters in scatter plots by city and state


```python
df.pivot_table(index=["search_city"], values=["salary"], \
               aggfunc=[np.mean,np.std,len]).plot.scatter(x="std",y="mean")
df.pivot_table(index=["search_city"], values=["salary"], \
               aggfunc=[np.mean,np.std,len]).plot.scatter(x="len",y="mean")
df.pivot_table(index=["search_city"], values=["reviews","ratings","salary"], \
               ).plot.scatter(x="ratings",y="salary")
df.pivot_table(index=["search_city"], values=["reviews","ratings","salary"], \
               ).plot.scatter(x="reviews",y="salary")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116713810>




![png](/images/project-4-analysis_files/project-4-analysis_18_1.png)



![png](/images/project-4-analysis_files/project-4-analysis_18_2.png)



![png](/images/project-4-analysis_files/project-4-analysis_18_3.png)



![png](/images/project-4-analysis_files/project-4-analysis_18_4.png)



```python
df.pivot_table(index=["state"], values=["salary"], \
               aggfunc=[np.mean,np.std,len]).plot.scatter(x="std",y="mean")
df.pivot_table(index=["state"], values=["salary"], \
               aggfunc=[np.mean,np.std,len]).plot.scatter(x="len",y="mean")
df.pivot_table(index=["state"], values=["reviews","ratings","salary"], \
               ).plot.scatter(x="ratings",y="salary")
df.pivot_table(index=["state"], values=["reviews","ratings","salary"], \
               ).plot.scatter(x="reviews",y="salary")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1118afad0>




![png](/images/project-4-analysis_files/project-4-analysis_19_1.png)



![png](/images/project-4-analysis_files/project-4-analysis_19_2.png)



![png](/images/project-4-analysis_files/project-4-analysis_19_3.png)



![png](/images/project-4-analysis_files/project-4-analysis_19_4.png)



```python
pd.scatter_matrix(df,alpha=0.1)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x11628d090>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11612d3d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116999410>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116a3ed50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1169c5e50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1118d00d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1165e6350>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116b4b790>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11686d810>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116a9d590>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x11684b4d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x115879910>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11628db10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1162f52d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116101bd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x116521c10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116477410>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1167f55d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x116430f50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1168d2250>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x11642bc10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1118eaf10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x115faffd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1162b8a90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1164bf910>]], dtype=object)




![png](/images/project-4-analysis_files/project-4-analysis_20_1.png)


### Define tiers of locations containing roughly 50 salary entries each


```python
tier_1 = ["Sherman Oaks, CA", "Overland Park, KS",
"Carlsbad, CA", "Phoenix, AZ",
"Raleigh, NC"]


tier_2 = ["Portland, OR", "Pittsburgh, PA",
"San Diego, CA", "St. Louis, MO",
"Marlborough, MA"]


tier_3 =["Salt Lake City, UT", "Newton, MA",
"St Petersburg Beach, FL",
"Deerfield Beach, FL",
"Port Washington, NY",
"Seattle, WA", "Berkeley, CA",
"Cupertino, CA", "Redmond, WA",
"Culver City, CA", "Santa Monica, CA"]


tier_4 = ["Houston, TX", "Denver, CO",
"Austin, TX", "Dallas, TX",
"Renton, WA", "Fort George G Meade, MD"]


tier_5 = ["Union, NJ", "Pasadena, CA",
"Los Angeles, CA", "Schaumburg, IL",
"Detroit, MI", "South Hackensack, NJ",
"Chevy Chase, MD", "Atlanta, GA"]

tier_6 = ["Torrance, CA", "Henderson, NV",
"Washington, DC", "Downers Grove, IL",
"Irvine, CA", "New York, NY",
"Mountain View, CA"]


tier_7 = ["Cambridge, MA", "San Jose, CA",
"South Plainfield, NJ",
"Chicago, IL"]


tier_8 = ["Jersey City, NJ", "Palo Alto, CA",
"Philadelphia, PA", "San Francisco, CA",
"McLean, VA", "Redwood City, CA",
"Boston, MA", "Manhattan, NY",
"San Ramon, CA", "Campbell, CA",
"Natick, MA", "San Mateo, CA"]
```


```python
df.insert(2,'tier',df['search_city'].map(lambda x: 1 if x in tier_1 else 2 if x in tier_2 else 3 if x in tier_3 \
                        else 4 if x in tier_4 else 5 if x in tier_5 else 6 if x in tier_6 \
                        else 7 if x in tier_7 else 8) )
```

### make variables for each location tier


```python
location_dummies = pd.get_dummies(df.tier,prefix="tier",drop_first=True)
```

## Below analysis should happen after train test split
- fit/generate on training
- apply to test

### make vairiables for most common words in each text field


```python
# fix locations by fplitting on comma and checking if length of second item is over 2
# make a new book for the analysis

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(
    binary=True,  # Create binary features
    stop_words='english', # Ignore common words such as 'the', 'and'
    max_features=50, # Only use the top 50 most common words
)

X_co = df[["company"]]
# This builds a matrix with a row per website (or data point) and column per word (using all words in the dataset)
X_co = v.fit_transform(df.company).todense()
X_co = pd.DataFrame(X_co, columns=["co_"+x for x in v.get_feature_names()])
# # print X_co.head()

X_ti = df[["jobtitle"]]
# This builds a matrix with a row per website (or data point) and column per word (using all words in the dataset)
X_ti = v.fit_transform(df.jobtitle).todense()
X_ti = pd.DataFrame(X_ti, columns=["ti_"+x for x in v.get_feature_names()])
# X_ti.head()


X_su = df[["jobtitle"]]
# This builds a matrix with a row per website (or data point) and column per word (using all words in the dataset)
X_su = v.fit_transform(df.jobtitle).todense()
X_su = pd.DataFrame(X_su, columns=["su_"+x for x in v.get_feature_names()])
# X_su.head()
```

## Build table for features


```python
mask = df["salary"].notnull()
X_train = pd.merge(df[mask]["days_ago_posted"].to_frame(),\
         pd.merge(X_co[mask], \
            pd.merge(X_ti[mask], \
                pd.merge(X_su[mask],location_dummies[mask],left_index=True,right_index=True), \
                    left_index=True,right_index=True), \
                 left_index=True,right_index=True), \
        left_index=True,right_index=True)

avg = df[mask].salary.mean()  # make separate values for each tier?
y_train = df[mask].salary.map(lambda x: 0 if x < avg*1.2 else 1)

mask = df["salary"].isnull()
X_pred = pd.merge(df[mask]["days_ago_posted"].to_frame(),\
         pd.merge(X_co[mask], \
            pd.merge(X_ti[mask], \
                pd.merge(X_su[mask],location_dummies[mask],left_index=True,right_index=True), \
                    left_index=True,right_index=True), \
                 left_index=True,right_index=True), \
        left_index=True,right_index=True)

y_pred = df[mask]
```


```python
X_train[X_train.isnull()].head()
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days_ago_posted</th>
      <th>co_aerospace</th>
      <th>co_allen</th>
      <th>co_america</th>
      <th>co_associates</th>
      <th>co_ball</th>
      <th>co_bank</th>
      <th>co_booz</th>
      <th>co_cancer</th>
      <th>co_capital</th>
      <th>...</th>
      <th>su_technical</th>
      <th>su_technician</th>
      <th>su_technologist</th>
      <th>tier_2</th>
      <th>tier_3</th>
      <th>tier_4</th>
      <th>tier_5</th>
      <th>tier_6</th>
      <th>tier_7</th>
      <th>tier_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>83</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>89</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>121</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>136</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 158 columns</p>
</div>




```python
# std_pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('scale', preprocessing.MaxAbsScaler()),
#     ('clf', LogisticRegression()),
# ])

# mm_pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('scale', preprocessing.StandardScaler()),
#     ('clf', LogisticRegression()),
# ])

# parameters = {
#     'vect__stop_words': ('english',),
#     'vect__binary': (True,False),
#     'vect__max_features': (10,25,50,75),
#     'clf__C': (0.00001, 0.001, 0.01, 0.1, 1, 2.5, 5, 7.5, 10, 25, 50, 100, 500, 1000, 10000),
#     'clf__penalty': ('l1','l2'),
#     'clf__solver': ('liblinear',),
# }


# grid_search = GridSearchCV(std_pipeline, parameters, verbose=False, cv=kf_shuffle)

# print("Performing grid search...")
# print("pipeline:", [name for name, _ in std_pipeline.steps])
# print("parameters:")
# print(parameters)

# grid_search.fit(X_train, y_train)

# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))


# # cv_pred = gs.predict(X_test)
```

# pipeline notes

use feature union to combine data
- requires key,value pairs to define proper data
- how do I flag data like this? can I do it in a dataframe??


```python
kf_shuffle = StratifiedKFold(n_splits=10,shuffle=True,random_state=7777)

model = make_pipeline(preprocessing.MinMaxScaler(), LogisticRegressionCV(penalty='l1',solver='liblinear'))
results = cross_val_score(model,X_train, y_train, cv=kf_shuffle)
print(results)

print results.mean(), results.std()
```

    [ 0.86046512  0.8372093   0.76744186  0.81395349  0.88372093  0.79069767
      0.81395349  0.79069767  0.82926829  0.90243902]
    0.828984685196 0.0407695010869



```python
model = make_pipeline(preprocessing.StandardScaler(), LogisticRegressionCV(penalty='l1',solver='liblinear'))
results = cross_val_score(model,X_train, y_train, cv=kf_shuffle)
print(results)

print results.mean(), results.std()
```

    [ 0.74418605  0.81395349  0.81395349  0.88372093  0.88372093  0.81395349
      0.81395349  0.79069767  0.82926829  0.85365854]
    0.824106636415 0.0399042487045


### repeat for l2 penalty


```python
model = make_pipeline(preprocessing.MinMaxScaler(), LogisticRegressionCV(penalty='l2',solver='liblinear'))
results = cross_val_score(model,X_train, y_train, cv=kf_shuffle)
print(results)

print results.mean(), results.std()
```

    [ 0.86046512  0.81395349  0.74418605  0.88372093  0.81395349  0.8372093
      0.81395349  0.81395349  0.85365854  0.92682927]
    0.836188315372 0.0467040828302



```python
model2 = make_pipeline(preprocessing.StandardScaler(), LogisticRegressionCV(penalty='l2',solver='liblinear'))
results2 = cross_val_score(model,X_train, y_train, cv=kf_shuffle)
print(results2)
print results2.mean(), results2.std()
```

    [ 0.86046512  0.81395349  0.74418605  0.88372093  0.81395349  0.8372093
      0.81395349  0.81395349  0.85365854  0.92682927]
    0.836188315372 0.0467040828302



```python
print results2.mean(), results2.std()
model2.fit(X_train, y_train)
```

    0.836188315372 0.0467040828302





    Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('logisticregressioncv', LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
               refit=True, scoring=None, solver='liblinear', tol=0.0001,
               verbose=0))])




```python
print results.mean(), results.std()
model.fit(X_train, y_train)
```

    0.836188315372 0.0467040828302





    Pipeline(steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('logisticregressioncv', LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
               refit=True, scoring=None, solver='liblinear', tol=0.0001,
               verbose=0))])




```python
print model.steps[1][1].C_
print model2.steps[1][1].C_
```

    [ 2.7825594]
    [ 0.00599484]



```python
print classification_report(y_train, model.predict(X_train))
pd.DataFrame(confusion_matrix(y_train, model.predict(X_train)),columns=["Low","High"],index=["Low","High"])
```

                 precision    recall  f1-score   support
    
              0       0.94      0.86      0.90       308
              1       0.71      0.86      0.77       118
    
    avg / total       0.88      0.86      0.87       426
    





<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Low</th>
      <th>High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Low</th>
      <td>266</td>
      <td>42</td>
    </tr>
    <tr>
      <th>High</th>
      <td>17</td>
      <td>101</td>
    </tr>
  </tbody>
</table>
</div>




```python
print classification_report(y_train, model2.predict(X_train))
print "Accuracy Scoreaccuracy_score(y_train, model2.predict(X_train))
pd.DataFrame(confusion_matrix(y_train, model2.predict(X_train)),columns=["Low","High"],index=["Low","High"])
```

                 precision    recall  f1-score   support
    
              0       0.94      0.86      0.90       308
              1       0.71      0.86      0.77       118
    
    avg / total       0.88      0.86      0.87       426
    
    0.861502347418





<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Low</th>
      <th>High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Low</th>
      <td>266</td>
      <td>42</td>
    </tr>
    <tr>
      <th>High</th>
      <td>17</td>
      <td>101</td>
    </tr>
  </tbody>
</table>
</div>




```python
## List of coefficients by importance
```


```python
coeffs = pd.DataFrame(model.steps[1][1].coef_, columns=X_train.columns,index=["coeff"]).T
abs_coeffs = abs(coeffs)
count = 1
for i in zip(abs_coeffs.sort_values("coeff",ascending=False).index, \
             abs_coeffs.sort_values("coeff",ascending=False).values):
    print("{:3d} {:16s}  {:.5f}").format(count, i[0], i[1][0])
    count += 1
```

      1 tier_8            0.19287
      2 ti_data           0.12159
      3 su_data           0.12159
      4 su_senior         0.11109
      5 ti_senior         0.11109
      6 ti_director       0.10379
      7 su_director       0.10379
      8 co_partners       0.10029
      9 co_university     0.09895
     10 co_corporation    0.09106
     11 co_technology     0.07903
     12 ti_machine        0.07271
     13 su_machine        0.07271
     14 su_principal      0.07196
     15 ti_principal      0.07196
     16 co_company        0.06868
     17 ti_learning       0.06814
     18 su_learning       0.06814
     19 days_ago_posted   0.06649
     20 ti_sr             0.06590
     21 su_sr             0.06590
     22 co_associates     0.06438
     23 tier_3            0.06411
     24 su_scientist      0.05775
     25 ti_scientist      0.05775
     26 tier_2            0.05087
     27 co_lawrence       0.04973
     28 co_llc            0.04825
     29 su_analyst        0.04809
     30 ti_analyst        0.04809
     31 tier_4            0.04668
     32 su_associate      0.04618
     33 ti_associate      0.04618
     34 tier_7            0.04608
     35 co_national       0.04481
     36 co_health         0.04476
     37 co_scientific     0.04187
     38 co_services       0.04000
     39 co_consulting     0.03798
     40 su_quantitative   0.03687
     41 ti_quantitative   0.03687
     42 su_developer      0.03661
     43 ti_developer      0.03661
     44 ti_analytics      0.03473
     45 su_analytics      0.03473
     46 ti_manager        0.03463
     47 su_manager        0.03463
     48 ti_statistician   0.03307
     49 su_statistician   0.03307
     50 su_research       0.03250
     51 ti_research       0.03250
     52 su_technician     0.03239
     53 ti_technician     0.03239
     54 ti_specialist     0.03146
     55 su_specialist     0.03146
     56 ti_process        0.03113
     57 su_process        0.03113
     58 co_group          0.03073
     59 co_washington     0.03013
     60 co_st             0.02854
     61 su_engineering    0.02742
     62 ti_engineering    0.02742
     63 su_programmer     0.02741
     64 ti_programmer     0.02741
     65 su_technical      0.02712
     66 ti_technical      0.02712
     67 co_honeywell      0.02690
     68 su_big            0.02671
     69 ti_big            0.02671
     70 su_marketing      0.02652
     71 ti_marketing      0.02652
     72 co_global         0.02625
     73 ti_intern         0.02622
     74 su_intern         0.02622
     75 ti_project        0.02576
     76 su_project        0.02576
     77 su_statistical    0.02542
     78 ti_statistical    0.02542
     79 su_computational  0.02470
     80 ti_computational  0.02470
     81 ti_analysis       0.02438
     82 su_analysis       0.02438
     83 su_analytical     0.02415
     84 ti_analytical     0.02415
     85 co_science        0.02374
     86 su_lead           0.02206
     87 ti_lead           0.02206
     88 co_solutions      0.02138
     89 co_laboratory     0.02109
     90 ti_medical        0.02106
     91 su_medical        0.02106
     92 ti_assistant      0.02081
     93 su_assistant      0.02081
     94 ti_ii             0.01995
     95 su_ii             0.01995
     96 su_market         0.01969
     97 ti_market         0.01969
     98 su_engineer       0.01880
     99 ti_engineer       0.01880
    100 ti_quality        0.01859
    101 su_quality        0.01859
    102 ti_systems        0.01787
    103 su_systems        0.01787
    104 ti_staff          0.01658
    105 su_staff          0.01658
    106 tier_6            0.01651
    107 su_clinical       0.01629
    108 ti_clinical       0.01629
    109 co_international  0.01622
    110 ti_technologist   0.01601
    111 su_technologist   0.01601
    112 su_operations     0.01531
    113 ti_operations     0.01531
    114 ti_lab            0.01382
    115 su_lab            0.01382
    116 co_research       0.01325
    117 co_technologies   0.01276
    118 co_management     0.01160
    119 su_business       0.01118
    120 ti_business       0.01118
    121 co_center         0.01068
    122 co_hospital       0.00889
    123 co_systems        0.00831
    124 su_science        0.00613
    125 ti_science        0.00613
    126 co_medical        0.00463
    127 su_iii            0.00344
    128 ti_iii            0.00344
    129 co_cancer         0.00341
    130 tier_5            0.00276
    131 ti_software       0.00244
    132 su_software       0.00244
    133 co_resources      0.00083
    134 ti_bioinformatics  0.00020
    135 su_bioinformatics  0.00020
    136 ti_development    0.00015
    137 su_development    0.00015
    138 co_general        0.00000
    139 co_hamilton       0.00000
    140 co_bank           0.00000
    141 co_institute      0.00000
    142 co_laboratories   0.00000
    143 co_leap           0.00000
    144 co_magic          0.00000
    145 ti_product        0.00000
    146 co_ball           0.00000
    147 co_microsoft      0.00000
    148 co_pfizer         0.00000
    149 co_pharmaceuticals  0.00000
    150 co_capital        0.00000
    151 co_aerospace      0.00000
    152 co_celgene        0.00000
    153 co_america        0.00000
    154 co_allen          0.00000
    155 co_tech           0.00000
    156 su_product        0.00000
    157 co_therapeutics   0.00000
    158 co_booz           0.00000



```python
abs_coeffs.sort_values("coeff",ascending=False).head(25).plot(kind="bar",figsize=(12,8),fontsize=16)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c9e2f50>




![png](/images/project-4-analysis_files/project-4-analysis_46_1.png)



```python

```


```python

```


```python
# model.score(X)
```

## now predict on other data


```python
model.predict(X)
```




    array([1, 1, 1, ..., 0, 1, 0])



### location groupings
- summarize cities into states where can
- group low salary places together
- geography??
- Tableau??


### Model optimizations:
- kNN
  + k
  + weights
- logreg
  + L1,L2
  + Cs
- scaling
  + MinMax
  + Standardized
- number of key words in each vectorized variable
  
pipelines vs function calls and where can I use gridsearch?

value of including reivews and ratings?


```python
df.search_city.nunique()
```




    82


