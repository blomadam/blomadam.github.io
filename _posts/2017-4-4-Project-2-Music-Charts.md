---
layout: post
title: Project 2 - Music Charts
---

This week I am looking at the Billboard top 100 most popular songs for 
roughly the entire year 2000.  As with any project, the steps to a good 
analysis are:

1. Understand the structure of your data set
2. Clean and transform your data set
3. Explore the data and form hypotheses
4. Conduct the analysis
5. Create the deliverables
6. Deliver your results


I am approaching this dataset from the mindset of a manager of an upcoming band.  I want to be able to advise them about how different genres perform in the charts.  Most notably, I am interested in the peak ranking, and also the duration tracks from each genre tend to stay in the charts.  With that in mind, let's get cracking!


## 1. Data structure

This dataset has 317 entries with 83 columns of data.  The first seven columns give details we are used to about the track: the year (which happens to be 2000 for every entry in this list), the artist name, track name, track length, and musical genre.  Then come the interesting data columns about each track's performance in the Billboard charts: the date each track first appeared on the chart, and the date each track peaked (reached its highest ranking), followed 76 columns storing each track's ranking for each week since it first entered the chart.

I did not question how any of these quantities were calculated, but simply charted the data as it appeared, while also doing a simple sanity check of researching a couple of the most popular songs of the year to make sure their performance matched my analysis of the data.  *Breathe* by Faith Hill, for example, I correctly identified as a massively popular song, but I see that its genre is listed as Rap.... though I, personally, would have considered it Country.  I see this exercise as vindication that I can correctly interpret the most important aspect of the data (i.e. the rankings), and caution should be taken before trusting data read from any table.

## 2. Data cleaning and transformation

At first glance, this dataset seemed remarkably complete.  Nearly every column was listed as completely full, and I thought just some simple type conversions would be required.  Careful reading of the `df.info()` result showed some trickery starting around column 30:
```
x24th.week         317 non-null object
x25th.week         39 non-null object
x26th.week         37 non-null object
x27th.week         30 non-null object
x28th.week         317 non-null object
```
Amidst the dozens of completely full columns lie three nearly empty ones!!  Further exploration shows:

```python
df.iloc[:,30:35].head().to_html()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x24th.week</th>
      <th>x25th.week</th>
      <th>x26th.week</th>
      <th>x27th.week</th>
      <th>x28th.week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>15</td>
      <td>22</td>
      <td>29</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>48</td>
      <td>47</td>
      <td>NaN</td>
      <td>*</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>14</td>
      <td>17</td>
      <td>21</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>*</td>
    </tr>
    <tr>
      <th>4</th>
      <td>*</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>*</td>
    </tr>
  </tbody>
</table>

