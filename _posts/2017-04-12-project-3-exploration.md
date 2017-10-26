---
layout: post
title: Housing market analysis
date: 2017-04-12
published: true
categories: projects
image: /images/project-3-exploration_files/project-3-exploration_28_1.png
---

We have a data set of real estate prices from Aimes, Iowa.  Our goal is to provide a model to predict prices using a subset of the columns to be able to reduce the size of the database the company is required to store.  We are also interested in providing descriptive statistics for the market, specifically which neighborhoods have many sales and high prices.  Finally, we are interested in describing how this data has changed over time.


## Results summary

The most significant predictor of housing price is the `OverallQuality` of the property.  Additionally, there is a nonlinear relation between this quality and `SalePrice`, with a quadratic term added to the model showing an even stronger correlation than the linear term.

I suggest a simplified model with six parameters to predict `SalePrices`.  The linear model with `['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea', 'OverallQualSquared']` explains the variation in the data just about as well as the best model with twice as many parameters (difference in R-squared on a test set is less than 0.01).  This parameter selection is stable over repeated selections of test and training data in a 2:1 split.

`NAmes` and `CollgCr` are the largest markets (with `CollgCr` also having high prices). `NoRidge`, `NridgHt`, and `StoneBr` have the highest average sale prices.  `Crawfor` has prices that are trending higher.


# Detailed Analysis

## Read in the data

First save the list of variables we are allowed to use, and read in the just the training data from those columns.


```python
avail_columns = ["LotArea", "Utilities", "Neighborhood", "BldgType", "HouseStyle", "OverallQual" \
           ,"OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl" \
           ,"GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr" \
           ,"MoSold", "YrSold", "SalePrice"]


df = pd.read_csv("train.csv", usecols=avail_columns)
print df
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>Utilities</th>
      <th>Neighborhood</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8450</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1710</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9600</td>
      <td>AllPub</td>
      <td>Veenker</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1262</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11250</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1786</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9550</td>
      <td>AllPub</td>
      <td>Crawfor</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14260</td>
      <td>AllPub</td>
      <td>NoRidge</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2198</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14115</td>
      <td>AllPub</td>
      <td>Mitchel</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>5</td>
      <td>1993</td>
      <td>1995</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1362</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>2009</td>
      <td>143000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10084</td>
      <td>AllPub</td>
      <td>Somerst</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1694</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>307000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10382</td>
      <td>AllPub</td>
      <td>NWAmes</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>6</td>
      <td>1973</td>
      <td>1973</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2090</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>11</td>
      <td>2009</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6120</td>
      <td>AllPub</td>
      <td>OldTown</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>7</td>
      <td>5</td>
      <td>1931</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1774</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2008</td>
      <td>129900</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7420</td>
      <td>AllPub</td>
      <td>BrkSide</td>
      <td>2fmCon</td>
      <td>1.5Unf</td>
      <td>5</td>
      <td>6</td>
      <td>1939</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1077</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2008</td>
      <td>118000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11200</td>
      <td>AllPub</td>
      <td>Sawyer</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1965</td>
      <td>1965</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1040</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2008</td>
      <td>129500</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11924</td>
      <td>AllPub</td>
      <td>NridgHt</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>9</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>2324</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>2006</td>
      <td>345000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12968</td>
      <td>AllPub</td>
      <td>Sawyer</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1962</td>
      <td>1962</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>912</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>2008</td>
      <td>144000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10652</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1494</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>279500</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10920</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1960</td>
      <td>1960</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1253</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>157000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6120</td>
      <td>AllPub</td>
      <td>BrkSide</td>
      <td>1Fam</td>
      <td>1.5Unf</td>
      <td>7</td>
      <td>8</td>
      <td>1929</td>
      <td>2001</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>854</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>2007</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>11241</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>7</td>
      <td>1970</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1004</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2010</td>
      <td>149000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10791</td>
      <td>AllPub</td>
      <td>Sawyer</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>1967</td>
      <td>1967</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1296</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>10</td>
      <td>2006</td>
      <td>90000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>13695</td>
      <td>AllPub</td>
      <td>SawyerW</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1114</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>2008</td>
      <td>159000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7560</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1958</td>
      <td>1965</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1339</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2009</td>
      <td>139000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>14215</td>
      <td>AllPub</td>
      <td>NridgHt</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2376</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>11</td>
      <td>2006</td>
      <td>325300</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7449</td>
      <td>AllPub</td>
      <td>IDOTRR</td>
      <td>1Fam</td>
      <td>1.5Unf</td>
      <td>7</td>
      <td>7</td>
      <td>1930</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1108</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>139400</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9742</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1795</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>2008</td>
      <td>230000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4224</td>
      <td>AllPub</td>
      <td>MeadowV</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1060</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>129900</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8246</td>
      <td>AllPub</td>
      <td>Sawyer</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>8</td>
      <td>1968</td>
      <td>2001</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1060</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2010</td>
      <td>154000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>14230</td>
      <td>AllPub</td>
      <td>NridgHt</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1600</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2009</td>
      <td>256300</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7200</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1951</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>900</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2010</td>
      <td>134800</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11478</td>
      <td>AllPub</td>
      <td>NridgHt</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1704</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2010</td>
      <td>306000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>16321</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1957</td>
      <td>1997</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1600</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>2006</td>
      <td>207500</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6324</td>
      <td>AllPub</td>
      <td>BrkSide</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>6</td>
      <td>1927</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>520</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>68500</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>1430</th>
      <td>21930</td>
      <td>AllPub</td>
      <td>Gilbert</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1838</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>2006</td>
      <td>192140</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>4928</td>
      <td>AllPub</td>
      <td>NPkVill</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>958</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2009</td>
      <td>143750</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>10800</td>
      <td>AllPub</td>
      <td>OldTown</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>6</td>
      <td>1927</td>
      <td>2007</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>968</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>64500</td>
    </tr>
    <tr>
      <th>1433</th>
      <td>10261</td>
      <td>AllPub</td>
      <td>Gilbert</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1792</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>186500</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>17400</td>
      <td>AllPub</td>
      <td>Mitchel</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1977</td>
      <td>1977</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1126</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2006</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>8400</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>9</td>
      <td>1962</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1537</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2008</td>
      <td>174000</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>9000</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>6</td>
      <td>1971</td>
      <td>1971</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>864</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2007</td>
      <td>120500</td>
    </tr>
    <tr>
      <th>1437</th>
      <td>12444</td>
      <td>AllPub</td>
      <td>NridgHt</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2008</td>
      <td>2008</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1932</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>11</td>
      <td>2008</td>
      <td>394617</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>7407</td>
      <td>AllPub</td>
      <td>OldTown</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>7</td>
      <td>1957</td>
      <td>1996</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1236</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2010</td>
      <td>149700</td>
    </tr>
    <tr>
      <th>1439</th>
      <td>11584</td>
      <td>AllPub</td>
      <td>NWAmes</td>
      <td>1Fam</td>
      <td>SLvl</td>
      <td>7</td>
      <td>6</td>
      <td>1979</td>
      <td>1979</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1725</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>11</td>
      <td>2007</td>
      <td>197000</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>11526</td>
      <td>AllPub</td>
      <td>Crawfor</td>
      <td>1Fam</td>
      <td>2.5Fin</td>
      <td>6</td>
      <td>7</td>
      <td>1922</td>
      <td>1994</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2555</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>2008</td>
      <td>191000</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>4426</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>848</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>149300</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>11003</td>
      <td>AllPub</td>
      <td>Somerst</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>10</td>
      <td>5</td>
      <td>2008</td>
      <td>2008</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2007</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2009</td>
      <td>310000</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>8854</td>
      <td>AllPub</td>
      <td>BrkSide</td>
      <td>1Fam</td>
      <td>1.5Unf</td>
      <td>6</td>
      <td>6</td>
      <td>1916</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>952</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2009</td>
      <td>121000</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>8500</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1422</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>11</td>
      <td>2007</td>
      <td>179600</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>8400</td>
      <td>AllPub</td>
      <td>Sawyer</td>
      <td>1Fam</td>
      <td>SFoyer</td>
      <td>6</td>
      <td>5</td>
      <td>1966</td>
      <td>1966</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>913</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2007</td>
      <td>129000</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>26142</td>
      <td>AllPub</td>
      <td>Mitchel</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1962</td>
      <td>1962</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1188</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2010</td>
      <td>157900</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>10000</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>1995</td>
      <td>1996</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2090</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>12</td>
      <td>2007</td>
      <td>240000</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>11767</td>
      <td>AllPub</td>
      <td>Edwards</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>4</td>
      <td>7</td>
      <td>1910</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1346</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2007</td>
      <td>112000</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>1533</td>
      <td>AllPub</td>
      <td>MeadowV</td>
      <td>Twnhs</td>
      <td>SFoyer</td>
      <td>5</td>
      <td>7</td>
      <td>1970</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>630</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>2006</td>
      <td>92000</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>9000</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>Duplex</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>1974</td>
      <td>1974</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1792</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>9</td>
      <td>2009</td>
      <td>136000</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>9262</td>
      <td>AllPub</td>
      <td>Somerst</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2008</td>
      <td>2009</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1578</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2009</td>
      <td>287090</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>3675</td>
      <td>AllPub</td>
      <td>Edwards</td>
      <td>TwnhsE</td>
      <td>SLvl</td>
      <td>5</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1072</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2006</td>
      <td>145000</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>17217</td>
      <td>AllPub</td>
      <td>Mitchel</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1140</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2006</td>
      <td>84500</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>7500</td>
      <td>AllPub</td>
      <td>Somerst</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1221</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2009</td>
      <td>185000</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>7917</td>
      <td>AllPub</td>
      <td>Gilbert</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1647</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>13175</td>
      <td>AllPub</td>
      <td>NWAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2073</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2010</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>9042</td>
      <td>AllPub</td>
      <td>Crawfor</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2340</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>2010</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>9717</td>
      <td>AllPub</td>
      <td>NAmes</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>1078</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2010</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>9937</td>
      <td>AllPub</td>
      <td>Edwards</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1256</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>2008</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 19 columns</p>
</div>



## Explore each variable and correlations

Check the info on the data frame for column data types and null values.  Any column with `dtype` of `int64` is complete since a single string or `NaN` entry would change the `dtype` for the entire column.  The object columns are all expected to have string values, so no worries there.  Check the value_counts on the object columns for mis-spelled labels.


```python
df['YrSold'].describe()
```
    count    1460.000000
    mean     2007.815753
    std         1.328095
    min      2006.000000
    25%      2007.000000
    50%      2008.000000
    75%      2009.000000
    max      2010.000000
    Name: YrSold, dtype: float64


```python
df.BldgType.value_counts()
```
    1Fam      1220
    TwnhsE     114
    Duplex      52
    Twnhs       43
    2fmCon      31
    Name: BldgType, dtype: int64



Need to decide which variables should become dummified.  We do this with categorical classifications that we expect to have a strong impact on the result.  Utilities and RoofMatl may not be good choices since there is so little variation... though we should inspect them to see if there is a strong correlation with the price.


```python
correl = df.corr()**2
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(correl);
```


![png](/images/project-3-exploration_files/project-3-exploration_18_0.png)



```python
for i in avail_columns:
    if df[i].dtype == int:
        df.plot(x=i,y="SalePrice",kind="scatter")
        plt.show()
```



![png](/images/project-3-exploration_files/project-3-exploration_19_1.png)



![png](/images/project-3-exploration_files/project-3-exploration_19_3.png)



![png](/images/project-3-exploration_files/project-3-exploration_19_4.png)



![png](/images/project-3-exploration_files/project-3-exploration_19_5.png)



![png](/images/project-3-exploration_files/project-3-exploration_19_6.png)




## Explore nonlinear relationships

Some of the features are correlated in a nonlinear fashion with the sales price, so I will check correlations with the logarithm of the sales price, and with the logarithm of the features...  I can also try polynomial fits with the features.  Overall Quality, YearBuilt, 

Looks like some features show outliers (such as LotArea being greather than 60000 units, HalfBath ==2, or BedroomAbvGr > 5, KitchenAbvGr == 0 or 3). One can make a stron case that the estimate for these lots should be given by a different function than for the others...  Perhaps these are times when the categorical variables would work well?


```python
plt.scatter(df.OverallQual**2,df.SalePrice)
#df.corr()
```

    <matplotlib.collections.PathCollection at 0x1178f5750>




![png](/images/project-3-exploration_files/project-3-exploration_21_1.png)



```python
df["OverallQualSquared"] = df.OverallQual**2
df["YearBuiltSquared"] = df.YearBuilt**2
```

## Training test split


```python
features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'GrLivArea', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'MoSold', 'YrSold',
       'OverallQualSquared', 'YearBuiltSquared']

X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
```

## Fitting


```python
# REGRESSION MODELS

results = []
for n in range(1,len(features)+1):
    max_r2 = 0
    best_features = []
    for cols in itertools.combinations(features, n):
        fit_xtrain = X_train[list(cols)]
        fit_xtest = X_test[list(cols)]        

#FIT LINEAR MODEL
        line = lm.LinearRegression(normalize=True)

#for algorithm in [line, lasso, ridge]:
        for algorithm in [line]:
            ## Fit model to data
            model = algorithm.fit(fit_xtrain,y_train)

            test_r2 = algorithm.score(fit_xtest, y_test)
#             print "Test r2 score : {}".format(test_r2);
            if test_r2 > max_r2:
                max_r2 = test_r2
                coeff_sum = abs(algorithm.coef_).mean()
                mse = mean_squared_error(y_true=y_test, y_pred=algorithm.predict(fit_xtest))
                best_features = list(cols)
    results.append( (max_r2,coeff_sum, mse, best_features))
results
```

    [(0.64493256596024184,  3738.6815185367077,  2186715449.1880145,
      ['OverallQualSquared']),
     (0.72915774869635741,  1444.2671637627586,  1668006914.8561254,
      ['GrLivArea', 'OverallQualSquared']),
     (0.75483471919634171,   939.9441940699586,  1509872930.4412842,
      ['YearBuilt', 'GrLivArea', 'OverallQualSquared']),
     (0.79589913657403477,  12011.917742913709,  1256973939.1988173,
      ['LotArea',       'OverallQual',       'OverallCond',
       'YearBuilt',       'GrLivArea',       'HalfBath',
       'KitchenAbvGr',       'OverallQualSquared']),
     (0.79640356966477699,  10782.082461790924,  1253867341.6152034,
      ['LotArea',       'OverallQual',       'OverallCond',
       'YearBuilt',       'GrLivArea',       'HalfBath',
       'BedroomAbvGr',       'KitchenAbvGr',       'OverallQualSquared']),
     (0.79670268601380689,  9925.112054464651,  1252025206.0690444,
      ['LotArea',       'OverallQual',       'OverallCond',
       'YearBuilt',       'GrLivArea',       'FullBath',
       'HalfBath',       'BedroomAbvGr',       'KitchenAbvGr',
       'OverallQualSquared']),
     (0.79624488667288296,  7607.3176169180288,  1254844605.4153514,
      ['LotArea',       'OverallQual',       'OverallCond',
       'YearBuilt',       'YearRemodAdd',       'GrLivArea',
       'FullBath',       'HalfBath',       'BedroomAbvGr',
       'KitchenAbvGr',       'MoSold',       'YrSold',
       'OverallQualSquared',       'YearBuiltSquared'])]




Plot the progression of the best coefficient of determination as the number of features selected grows.

```python
plt.plot(range(1,len(results)+1),[r[0] for r in results])
plt.xlabel("number of features")
plt.ylabel("best R^2")
```

![png](/images/project-3-exploration_files/project-3-exploration_27_1.png)

Plot the absolute sum of the coefficients.  Note that this only works when the features are normalized in some fashion before fitting.  A significant increase in this sum is an indication of the model being overfit.  Comparing this to the difference in R2 score would make a good cross check for overfitting.

```python
plt.plot(range(1,len(results)+1),[r[1] for r in results])
plt.xlabel("number of features")
plt.ylabel("Abs sum of coeffs")
```

![png](/images/project-3-exploration_files/project-3-exploration_28_1.png)


Plot the best mean squared error (MSE) for each number of features corresponding to the best R2 score.

```python
plt.plot(range(1,len(results)+1),[r[2] for r in results])
plt.xlabel("number of features")
plt.ylabel("MSE")
```

![png](/images/project-3-exploration_files/project-3-exploration_29_1.png)


Compare the be R2 score with a value from near the point of diminishing R2 returns.

```python
print max([r[0] for r in results])
results[5]
```

    0.796794359218


    (0.79010957401318027,     12257.811452486383,     1292629492.7138808,
     ['LotArea',      'OverallQual',      'OverallCond',
      'YearBuilt',      'GrLivArea',      'OverallQualSquared'])



## Selected model

The r-squared value for the test set drops plateaus around 0.83, when six features are used.  The set selected is: ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea', 'OverallQualSquared'].  The linear model with these features selected from the entire training set will capture most of the trends in the data without risking overfitting or storing extraneous data (particualrly since one of the features selected is derived from another!).

The model with 13 parameters (excluding only month sold) consistently beats the model with 6 parameters, but only by a small margin.  Considering the colinearity between many of the parameters (seen in the heatmap during initial data exploration) a smaller parameter set is desireable.  Selecting parameters with Ridge and Lasso was not as successful as the fitting with limited parameter sets, so I have opted for the latter.

A model with 8 parameters sometimes did slightly better than the model with 6, but the feature selection was not consistent (due to colinearity), so I went for the six final parameters.  The differences in R^2 between the full model and 6 parameters were someitmes as large as 0.01, while the 8 parameter fit was half that.  This increase in success did not seem to justify the loss in certainty over the best feature set.


```python
correl = df[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea', 'OverallQualSquared','SalePrice']].corr()**2
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(correl);
```


![png](/images/project-3-exploration_files/project-3-exploration_32_0.png)


## Neighborhood summary

The first figure shows the distribution of sales prices.  Anything over $170k will be toward the high end of the market.  Anything over $300k is really an exclusive property.

Below that is a box plot graph showing the distribution of prices in each neighborhood.  It seems NoRidge, NridgHt, and StoneBr have higher median SalePrice than the other neighborhoods.  A t test on this hypothesis shows significance with a p-value less than 0.0001.  

Below that are tables showing the number of homes sold and the mean sale price in each neighborhood.  Interestingly, there is no correlation between those two variables (as seen by the scatter plot below the tables.

NridgHt, in particular, has a good combination of sale price and sale volume.  NAmes and CollgCr are two other neighborhoods that have lower home prices, but much more sales volume.  CollgCr is a good market with over 100 homes sold in the dataset, and a mean sale price of nearly $200k.

Finally, I have some charts and tables showing how sale price and volume change over time in each neighborhood in the data set.  This is a fairly stable market in terms of meaningful rankings of the neighborhoods.  Crawfor, however, is worth watching, as it's mean sale price has been trending upward during the duration of the data set.

To sum up: NAmes and CollgCr are the largest markets (with CollgCr also having high prices). NoRidge, NridgHt, and StoneBr have the highest average sale prices.  Crawfor has prices that are trending higher.


```python
df.SalePrice.hist(bins=20)
```

    <matplotlib.axes._subplots.AxesSubplot at 0x1178356d0>


![png](/images/project-3-exploration_files/project-3-exploration_34_1.png)



```python
df.boxplot(column="SalePrice", by="Neighborhood",figsize=(14,8),rot=90,fontsize=16)
```

    <matplotlib.axes._subplots.AxesSubplot at 0x1182aa550>


![png](/images/project-3-exploration_files/project-3-exploration_35_1.png)



```python
mask = (df['Neighborhood'] =='NoRidge' ) \
        | (df['Neighborhood']=='NridgHt') \
        | (df['Neighborhood']=='StoneBr')
tstat, pval = stats.ttest_ind(a= df[mask]['SalePrice'], \
                b= df[[not x for x in mask]]['SalePrice'], \
                equal_var=False)
print 2* pval
```

    6.94365699791e-37



```python
df.pivot_table(index=["Neighborhood"], values=["SalePrice"], \
               aggfunc=[np.mean,len]).sort_values(('mean','SalePrice'),ascending=False)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>mean</th>
      <th>len</th>
    </tr>
    <tr>
      <th></th>
      <th>SalePrice</th>
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NoRidge</th>
      <td>335295</td>
      <td>41</td>
    </tr>
    <tr>
      <th>NridgHt</th>
      <td>316270</td>
      <td>77</td>
    </tr>
    <tr>
      <th>StoneBr</th>
      <td>310499</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Timber</th>
      <td>242247</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Veenker</th>
      <td>238772</td>
      <td>11</td>
    </tr>
    <tr>
      <th>Somerst</th>
      <td>225379</td>
      <td>86</td>
    </tr>
    <tr>
      <th>ClearCr</th>
      <td>212565</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Crawfor</th>
      <td>210624</td>
      <td>51</td>
    </tr>
    <tr>
      <th>CollgCr</th>
      <td>197965</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Blmngtn</th>
      <td>194870</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Gilbert</th>
      <td>192854</td>
      <td>79</td>
    </tr>
    <tr>
      <th>NWAmes</th>
      <td>189050</td>
      <td>73</td>
    </tr>
    <tr>
      <th>SawyerW</th>
      <td>186555</td>
      <td>59</td>
    </tr>
    <tr>
      <th>Mitchel</th>
      <td>156270</td>
      <td>49</td>
    </tr>
    <tr>
      <th>NAmes</th>
      <td>145847</td>
      <td>225</td>
    </tr>
    <tr>
      <th>NPkVill</th>
      <td>142694</td>
      <td>9</td>
    </tr>
    <tr>
      <th>SWISU</th>
      <td>142591</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Blueste</th>
      <td>137500</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Sawyer</th>
      <td>136793</td>
      <td>74</td>
    </tr>
    <tr>
      <th>OldTown</th>
      <td>128225</td>
      <td>113</td>
    </tr>
    <tr>
      <th>Edwards</th>
      <td>128219</td>
      <td>100</td>
    </tr>
    <tr>
      <th>BrkSide</th>
      <td>124834</td>
      <td>58</td>
    </tr>
    <tr>
      <th>BrDale</th>
      <td>104493</td>
      <td>16</td>
    </tr>
    <tr>
      <th>IDOTRR</th>
      <td>100123</td>
      <td>37</td>
    </tr>
    <tr>
      <th>MeadowV</th>
      <td>98576</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot_table(index=["Neighborhood"], values=["SalePrice"], \
               aggfunc=[np.mean,len]).sort_values(('len','SalePrice'),ascending=False)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>mean</th>
      <th>len</th>
    </tr>
    <tr>
      <th></th>
      <th>SalePrice</th>
      <th>SalePrice</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NAmes</th>
      <td>145847</td>
      <td>225</td>
    </tr>
    <tr>
      <th>CollgCr</th>
      <td>197965</td>
      <td>150</td>
    </tr>
    <tr>
      <th>OldTown</th>
      <td>128225</td>
      <td>113</td>
    </tr>
    <tr>
      <th>Edwards</th>
      <td>128219</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Somerst</th>
      <td>225379</td>
      <td>86</td>
    </tr>
    <tr>
      <th>Gilbert</th>
      <td>192854</td>
      <td>79</td>
    </tr>
    <tr>
      <th>NridgHt</th>
      <td>316270</td>
      <td>77</td>
    </tr>
    <tr>
      <th>Sawyer</th>
      <td>136793</td>
      <td>74</td>
    </tr>
    <tr>
      <th>NWAmes</th>
      <td>189050</td>
      <td>73</td>
    </tr>
    <tr>
      <th>SawyerW</th>
      <td>186555</td>
      <td>59</td>
    </tr>
    <tr>
      <th>BrkSide</th>
      <td>124834</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Crawfor</th>
      <td>210624</td>
      <td>51</td>
    </tr>
    <tr>
      <th>Mitchel</th>
      <td>156270</td>
      <td>49</td>
    </tr>
    <tr>
      <th>NoRidge</th>
      <td>335295</td>
      <td>41</td>
    </tr>
    <tr>
      <th>Timber</th>
      <td>242247</td>
      <td>38</td>
    </tr>
    <tr>
      <th>IDOTRR</th>
      <td>100123</td>
      <td>37</td>
    </tr>
    <tr>
      <th>ClearCr</th>
      <td>212565</td>
      <td>28</td>
    </tr>
    <tr>
      <th>StoneBr</th>
      <td>310499</td>
      <td>25</td>
    </tr>
    <tr>
      <th>SWISU</th>
      <td>142591</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Blmngtn</th>
      <td>194870</td>
      <td>17</td>
    </tr>
    <tr>
      <th>MeadowV</th>
      <td>98576</td>
      <td>17</td>
    </tr>
    <tr>
      <th>BrDale</th>
      <td>104493</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Veenker</th>
      <td>238772</td>
      <td>11</td>
    </tr>
    <tr>
      <th>NPkVill</th>
      <td>142694</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Blueste</th>
      <td>137500</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot_table(index=["Neighborhood"], values=["SalePrice"], \
               aggfunc=[np.mean,len]).plot.scatter(x="mean",y="len")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x118173610>




![png](/images/project-3-exploration_files/project-3-exploration_39_1.png)



```python
df.pivot_table(index=["YrSold"], values=["SalePrice"], \
               columns=["Neighborhood"],aggfunc=np.mean \
              ).plot(figsize=(12,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x117bace50>




![png](/images/project-3-exploration_files/project-3-exploration_40_1.png)



```python
df.pivot_table(index=["Neighborhood"], values=["SalePrice"], \
               columns=["YrSold"],aggfunc=np.mean \
              ).sort_values(("SalePrice",2010),ascending=False)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">SalePrice</th>
    </tr>
    <tr>
      <th>YrSold</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>StoneBr</th>
      <td>365046.0</td>
      <td>279585.0</td>
      <td>245000.0</td>
      <td>319967.0</td>
      <td>318886.0</td>
    </tr>
    <tr>
      <th>NridgHt</th>
      <td>305491.0</td>
      <td>310833.0</td>
      <td>332422.0</td>
      <td>323143.0</td>
      <td>308281.0</td>
    </tr>
    <tr>
      <th>Crawfor</th>
      <td>196635.0</td>
      <td>198777.0</td>
      <td>254411.0</td>
      <td>180211.0</td>
      <td>296833.0</td>
    </tr>
    <tr>
      <th>NoRidge</th>
      <td>322333.0</td>
      <td>399730.0</td>
      <td>304750.0</td>
      <td>323875.0</td>
      <td>289938.0</td>
    </tr>
    <tr>
      <th>ClearCr</th>
      <td>199166.0</td>
      <td>236333.0</td>
      <td>208991.0</td>
      <td>169875.0</td>
      <td>246850.0</td>
    </tr>
    <tr>
      <th>Timber</th>
      <td>264485.0</td>
      <td>229470.0</td>
      <td>234361.0</td>
      <td>245437.0</td>
      <td>245160.0</td>
    </tr>
    <tr>
      <th>Somerst</th>
      <td>210268.0</td>
      <td>233248.0</td>
      <td>225631.0</td>
      <td>236315.0</td>
      <td>206762.0</td>
    </tr>
    <tr>
      <th>CollgCr</th>
      <td>199016.0</td>
      <td>213999.0</td>
      <td>187718.0</td>
      <td>192317.0</td>
      <td>203700.0</td>
    </tr>
    <tr>
      <th>Blmngtn</th>
      <td>217087.0</td>
      <td>183350.0</td>
      <td>175447.0</td>
      <td>176720.0</td>
      <td>192000.0</td>
    </tr>
    <tr>
      <th>NWAmes</th>
      <td>199463.0</td>
      <td>175267.0</td>
      <td>193820.0</td>
      <td>185133.0</td>
      <td>187428.0</td>
    </tr>
    <tr>
      <th>Gilbert</th>
      <td>200250.0</td>
      <td>181967.0</td>
      <td>186000.0</td>
      <td>199955.0</td>
      <td>185500.0</td>
    </tr>
    <tr>
      <th>SawyerW</th>
      <td>164787.0</td>
      <td>209300.0</td>
      <td>184080.0</td>
      <td>183934.0</td>
      <td>184076.0</td>
    </tr>
    <tr>
      <th>Mitchel</th>
      <td>150036.0</td>
      <td>136731.0</td>
      <td>165280.0</td>
      <td>167860.0</td>
      <td>166950.0</td>
    </tr>
    <tr>
      <th>NAmes</th>
      <td>138985.0</td>
      <td>142962.0</td>
      <td>151553.0</td>
      <td>143880.0</td>
      <td>153665.0</td>
    </tr>
    <tr>
      <th>SWISU</th>
      <td>130125.0</td>
      <td>187500.0</td>
      <td>139612.0</td>
      <td>141048.0</td>
      <td>141333.0</td>
    </tr>
    <tr>
      <th>NPkVill</th>
      <td>NaN</td>
      <td>141500.0</td>
      <td>140000.0</td>
      <td>146937.0</td>
      <td>136750.0</td>
    </tr>
    <tr>
      <th>Sawyer</th>
      <td>149735.0</td>
      <td>133935.0</td>
      <td>128900.0</td>
      <td>136925.0</td>
      <td>132400.0</td>
    </tr>
    <tr>
      <th>OldTown</th>
      <td>135963.0</td>
      <td>114794.0</td>
      <td>147670.0</td>
      <td>116378.0</td>
      <td>122464.0</td>
    </tr>
    <tr>
      <th>Edwards</th>
      <td>134403.0</td>
      <td>132588.0</td>
      <td>132473.0</td>
      <td>123855.0</td>
      <td>111445.0</td>
    </tr>
    <tr>
      <th>BrkSide</th>
      <td>112746.0</td>
      <td>135737.0</td>
      <td>121707.0</td>
      <td>134994.0</td>
      <td>96500.0</td>
    </tr>
    <tr>
      <th>BrDale</th>
      <td>96750.0</td>
      <td>113833.0</td>
      <td>95225.0</td>
      <td>118625.0</td>
      <td>88000.0</td>
    </tr>
    <tr>
      <th>IDOTRR</th>
      <td>95758.0</td>
      <td>118933.0</td>
      <td>91642.0</td>
      <td>89580.0</td>
      <td>86278.0</td>
    </tr>
    <tr>
      <th>MeadowV</th>
      <td>123466.0</td>
      <td>105850.0</td>
      <td>98000.0</td>
      <td>88400.0</td>
      <td>81333.0</td>
    </tr>
    <tr>
      <th>Blueste</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>151000.0</td>
      <td>124000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Veenker</th>
      <td>273333.0</td>
      <td>214900.0</td>
      <td>244000.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot_table(index=["YrSold"], values=["SalePrice"], columns=["Neighborhood"],aggfunc=len \
              ).plot(figsize=(12,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x117da6f90>




![png](/images/project-3-exploration_files/project-3-exploration_42_1.png)

