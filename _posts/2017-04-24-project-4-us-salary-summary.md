---
layout: post
title: US Salary Summary
date: 2017-04-23
published: true
categories: projects
image: /images/project4/Size.png
---


Indeed has published Data Scientist salary information on [their site](https://www.indeed.com/salaries/Data-Scientist-Salaries).  The publish data averaged over the entire country, for many (~35) individual states, and for several cities in each of those states.  I am looping over each page in their location list to collect information on the salary average and range and an indication of the size of the market in each location.


```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.indeed.com/salaries/Data-Scientist-Salaries"

res = requests.get(URL)
page_source = BeautifulSoup(res.content,"lxml")
```


### Example market size extraction

```python
USA_respondents = int(page_source.find('div', class_="cmp-salary-header-content").text.split()[3].replace(",",""))
# this is the number of salary survey respondents
# which I will use as an indicator of size of industry in the location
print("salary info collected from {:,d} entities in the USA").format(USA_respondents)
```

    salary info collected from 35,741 entities in the USA


### Extract lists of URLs for states and cities


```python
state_list = page_source.findAll('option', {'data-tn-element':"loc_state[]"})
state_URLs = [i.attrs['value'] for i in state_list]
state_URLs[:3]
```

    ['/salaries/Data-Scientist-Salaries,-Arizona',
     '/salaries/Data-Scientist-Salaries,-Arkansas',
     '/salaries/Data-Scientist-Salaries,-California']


```python
city_list = page_source.findAll('option', {'data-tn-element':"loc_city[]"})
city_URLs = [i.attrs['value'] for i in city_list]
city_URLs[:13]  # to show several states
```




    ['/salaries/Data-Scientist-Salaries,-Chandler-AZ',
     '/salaries/Data-Scientist-Salaries,-Peoria-AZ',
     '/salaries/Data-Scientist-Salaries,-Phoenix-AZ',
     '/salaries/Data-Scientist-Salaries,-Scottsdale-AZ',
     '/salaries/Data-Scientist-Salaries,-Bentonville-AR',
     '/salaries/Data-Scientist-Salaries,-Little-Rock-AR',
     '/salaries/Data-Scientist-Salaries,-Aliso-Viejo-CA',
     '/salaries/Data-Scientist-Salaries,-Belmont-CA',
     '/salaries/Data-Scientist-Salaries,-Berkeley-CA',
     '/salaries/Data-Scientist-Salaries,-Beverly-Hills-CA',
     '/salaries/Data-Scientist-Salaries,-Brisbane-CA',
     '/salaries/Data-Scientist-Salaries,-Burlingame-CA',
     '/salaries/Data-Scientist-Salaries,-Campbell-CA']



### Example salary info extraction


```python
USA_min_salary = page_source.find('div', class_="cmp-sal-min").text
print("{} is the minimum reported salary in the USA").format(USA_min_salary)
```

    $45,000 is the minimum reported salary in the USA



```python
USA_max_salary = page_source.find('div', class_="cmp-sal-max").text
print("{} is the maximum reported salary in the USA").format(USA_max_salary)
```

    $257,000 is the maximum reported salary in the USA



```python
USA_avg_salary = page_source.find('div', class_="cmp-sal-salary").text.encode('utf-8')
print("{} is the average reported salary in the USA").format(USA_avg_salary)
```

    $129,996 per year is the average reported salary in the USA


## Generalize examples into functions for use in loop


```python
def extract_min_salary(location_source):
    a = location_source.find('div', class_="cmp-sal-min")
    return a.text.strip() if a else None

def extract_max_salary(location_source):
    a = location_source.find('div', class_="cmp-sal-max")
    return a.text.strip() if a else None

def extract_avg_salary(location_source):
    a = location_source.find('div', class_="cmp-sal-salary")
    return a.text.strip() if a else None

def extract_respondents(location_source):
    a = location_source.find('div', class_="cmp-salary-header-content")
    return a.text.split()[3] if a else None
```

## Loop over states


```python
state_results = []
for loc in state_URLs:
    state_res = requests.get("https://www.indeed.com"+loc)
    state_source = BeautifulSoup(state_res.content,"lxml")
    state_results.append(
                        (" ".join(loc.split("-")[3:]), \
                        extract_respondents(state_source), \
                        extract_min_salary(state_source), \
                        extract_max_salary(state_source), \
                        extract_avg_salary(state_source))
                        )
```


```python
state_results
```




    [('Arizona', u'78', u'$37,000', u'$293,000', u'$134,893\xa0per year'),
     ('Arkansas', u'19', u'$42,000', u'$211,000', u'$110,310\xa0per year'),
     ('California', u'13,556', u'$53,000', u'$271,000', u'$141,132\xa0per year'),
     ('Colorado', u'473', u'$44,000', u'$199,000', u'$107,772\xa0per year'),
     ('Connecticut', u'105', u'$41,000', u'$214,000', u'$110,914\xa0per year'),
     ... abbreviated ...
     ('Washington State',
      u'1,216',
      u'$51,000',
      u'$244,000',
      u'$130,428\xa0per year'),
     ('Wisconsin', u'119', u'$40,000', u'$266,000', u'$128,591\xa0per year')]




```python
city_results = []
for loc in city_URLs:
    city_res = requests.get("https://www.indeed.com"+loc)
    city_source = BeautifulSoup(city_res.content,"lxml")
    city_results.append(
                        (" ".join(loc.split("-")[3:-1]),loc.split("-")[-1], \
                        extract_respondents(city_source), \
                        extract_min_salary(city_source), \
                        extract_max_salary(city_source), \
                        extract_avg_salary(city_source))
                        )
```


```python
city_results
```




    [('Chandler', 'AZ', u'6', u'$58,000', u'$189,000', u'$115,171\xa0per year'),
     ('Peoria', 'AZ', u'7', u'$26.25', u'$78.75', u'$52.44\xa0per hour'),
     ('Phoenix', 'AZ', u'45', u'$45,000', u'$329,000', u'$154,611\xa0per year'),
     ('Scottsdale', 'AZ', u'20', u'$44,000', u'$223,000', u'$116,825\xa0per year'),
     ('Bentonville', 'AR', u'10', u'$44,000', u'$167,000', u'$95,697\xa0per year'),
     ('Little Rock', 'AR', u'6', u'$56,000', u'$224,000', u'$126,671\xa0per year'),
     ('Aliso Viejo', 'CA', u'8', u'$59,000', u'$178,000', u'$117,018\xa0per year'),
     ('Belmont', 'CA', u'12', u'$43,000', u'$334,000', u'$154,806\xa0per year'),
     ('Berkeley', 'CA', u'520', u'$46,000', u'$300,000', u'$146,102\xa0per year'),
     ('Beverly Hills',
      'CA',
      u'25',
      u'$56,000',
      u'$226,000',
      u'$127,280\xa0per year'),
     ('Brisbane', 'CA', u'38', u'$31,000', u'$214,000', u'$102,626\xa0per year'),
     ('Burlingame', 'CA', u'9', u'$61,000', u'$197,000', u'$121,329\xa0per year'),
     ('Campbell', 'CA', u'18', u'$38,000', u'$315,000', u'$143,933\xa0per year'),
     ('Carlsbad', 'CA', u'149', u'$47,000', u'$215,000', u'$116,523\xa0per year'),
     ... abbreviated ...
     ('Vancouver', 'WA', u'6', u'$65,000', u'$196,000', u'$128,488\xa0per year'),
     ('Kohler', 'WI', u'5', u'$70,000', u'$210,000', u'$140,000\xa0per year'),
     ('Madison', 'WI', u'76', u'$41,000', u'$268,000', u'$130,306\xa0per year'),
     ('Milwaukee', 'WI', u'34', u'$37,000', u'$271,000', u'$127,714\xa0per year')]



## Convert results to DataFrames


```python
state_df = pd.DataFrame(state_results, columns=('state','number_resps',"min_salary","max_salary",'avg_salary'))
city_df = pd.DataFrame(city_results, columns=('city','State','number_resps',"min_salary","max_salary",'avg_salary'))
print state_df.head(2)
print
print city_df.head(2)
```

          state number_resps min_salary max_salary         avg_salary
    0   Arizona           78    $37,000   $293,000  $134,893 per year
    1  Arkansas           19    $42,000   $211,000  $110,310 per year
    
           city State number_resps min_salary max_salary         avg_salary
    0  Chandler    AZ            6    $58,000   $189,000  $115,171 per year
    1    Peoria    AZ            7     $26.25     $78.75    $52.44 per hour



```python
city_df.head(5)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>State</th>
      <th>number_resps</th>
      <th>min_salary</th>
      <th>max_salary</th>
      <th>avg_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chandler</td>
      <td>AZ</td>
      <td>6</td>
      <td>$58,000</td>
      <td>$189,000</td>
      <td>$115,171 per year</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peoria</td>
      <td>AZ</td>
      <td>7</td>
      <td>$26.25</td>
      <td>$78.75</td>
      <td>$52.44 per hour</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>45</td>
      <td>$45,000</td>
      <td>$329,000</td>
      <td>$154,611 per year</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Scottsdale</td>
      <td>AZ</td>
      <td>20</td>
      <td>$44,000</td>
      <td>$223,000</td>
      <td>$116,825 per year</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bentonville</td>
      <td>AR</td>
      <td>10</td>
      <td>$44,000</td>
      <td>$167,000</td>
      <td>$95,697 per year</td>
    </tr>
  </tbody>
</table>
</div>



# Function to fix salaries


```python
def fix_salaries(salary_string):
    """
    takes salary string from indeed listing 
    and converts it to annual equivalient.
    """
    salary = 0
    salary_list = salary_string.replace("$","").replace(",","").strip().split()
    if "-" in salary_list[0]:
        temp = salary_list[0].split("-")
        salary = sum([float(a) for a in temp])/len(temp)
    else:
        salary = float(salary_list[0])
    if salary_list[-1] == "month":
        salary *= 12
    elif salary_list[-1] == "hour":
        salary *= 2000
    return salary
    
```


```python
city_df['avg_salary'] = city_df['avg_salary'].map(fix_salaries)
state_df['avg_salary'] = state_df['avg_salary'].map(fix_salaries)
```

## Silicone Valley and NYC have the highest salaries on average


```python
cities_money = city_df.sort_values(by='avg_salary',ascending=False).head(50)
cities_money.head(10)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>State</th>
      <th>number_resps</th>
      <th>min_salary</th>
      <th>max_salary</th>
      <th>avg_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>168</th>
      <td>South Plainfield</td>
      <td>NJ</td>
      <td>81</td>
      <td>$93,000</td>
      <td>$347,000</td>
      <td>200233.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Los Gatos</td>
      <td>CA</td>
      <td>105</td>
      <td>$78,000</td>
      <td>$359,000</td>
      <td>193215.0</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Roland</td>
      <td>OK</td>
      <td>5</td>
      <td>$95,000</td>
      <td>$289,000</td>
      <td>187624.0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Manhattan</td>
      <td>NY</td>
      <td>169</td>
      <td>$76,000</td>
      <td>$309,000</td>
      <td>173056.0</td>
    </tr>
    <tr>
      <th>167</th>
      <td>South Hackensack</td>
      <td>NJ</td>
      <td>6</td>
      <td>$67,000</td>
      <td>$322,000</td>
      <td>171467.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Marina del Rey</td>
      <td>CA</td>
      <td>119</td>
      <td>$85,000</td>
      <td>$268,000</td>
      <td>167764.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>South San Francisco</td>
      <td>CA</td>
      <td>15</td>
      <td>$57,000</td>
      <td>$328,000</td>
      <td>165416.0</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Philadelphia</td>
      <td>NY</td>
      <td>20</td>
      <td>$80,000</td>
      <td>$243,000</td>
      <td>160839.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>West Hollywood</td>
      <td>CA</td>
      <td>49</td>
      <td>$44,000</td>
      <td>$343,000</td>
      <td>158234.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>San Francisco Bay Area</td>
      <td>CA</td>
      <td>30</td>
      <td>$73,000</td>
      <td>$272,000</td>
      <td>156986.0</td>
    </tr>
  </tbody>
</table>
</div>



# Fix the number of respondents


```python
city_df['number_resps'] = city_df['number_resps'].map(lambda x: x.replace(",","")).map(int)
state_df['number_resps'] = state_df['number_resps'].map(lambda x: x.replace(",","")).map(int)
```

## SF, SV, NYC, Boston, and Chicago are all major markets


```python
cities_jobs = city_df.sort_values(by='number_resps',ascending=False).head(50)
cities_jobs.head(10)
```




<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>State</th>
      <th>number_resps</th>
      <th>min_salary</th>
      <th>max_salary</th>
      <th>avg_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>San Francisco</td>
      <td>CA</td>
      <td>4488</td>
      <td>$50,000</td>
      <td>$274,000</td>
      <td>140286.0</td>
    </tr>
    <tr>
      <th>175</th>
      <td>New York</td>
      <td>NY</td>
      <td>4306</td>
      <td>$45,000</td>
      <td>$284,000</td>
      <td>139598.0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Cambridge</td>
      <td>MA</td>
      <td>2093</td>
      <td>$62,000</td>
      <td>$228,000</td>
      <td>132209.0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Boston</td>
      <td>MA</td>
      <td>1955</td>
      <td>$53,000</td>
      <td>$211,000</td>
      <td>119223.0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Chicago</td>
      <td>IL</td>
      <td>1349</td>
      <td>$54,000</td>
      <td>$229,000</td>
      <td>126485.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Palo Alto</td>
      <td>CA</td>
      <td>1127</td>
      <td>$61,000</td>
      <td>$264,000</td>
      <td>145239.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>San Jose</td>
      <td>CA</td>
      <td>819</td>
      <td>$65,000</td>
      <td>$256,000</td>
      <td>144811.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Redwood City</td>
      <td>CA</td>
      <td>722</td>
      <td>$67,000</td>
      <td>$279,000</td>
      <td>155545.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Washington</td>
      <td>DC</td>
      <td>721</td>
      <td>$38,000</td>
      <td>$244,000</td>
      <td>119036.0</td>
    </tr>
    <tr>
      <th>131</th>
      <td>Marlborough</td>
      <td>MA</td>
      <td>721</td>
      <td>$65,000</td>
      <td>$198,000</td>
      <td>129984.0</td>
    </tr>
  </tbody>
</table>
</div>



## Make list of the top cities to search
Copy these results over to next file in analysis path.


```python
cities_to_search = pd.concat([cities_jobs,cities_money])
cities_to_search.drop_duplicates(inplace=True)
[x[0].replace(" ","+") + "%2C+" + x[1] for x in list(cities_to_search[['city','State']].values)]
```




    ['San+Francisco%2C+CA',
     'New+York%2C+NY',
     'Cambridge%2C+MA',
     'Boston%2C+MA',
     'Chicago%2C+IL',
     'Palo+Alto%2C+CA',
     'San+Jose%2C+CA',
     'Redwood+City%2C+CA',
     .... abbreviated ....
     'Renton%2C+WA',
     'Portland%2C+OR',
     'San+Francisco%2C+CA',
     'St+Petersburg+Beach%2C+FL',
     'Kohler%2C+WI',
     'Foster+City%2C+CA',
     'New+York%2C+NY',
     'Union%2C+NJ']



# Write results to file for other analysis


```python
state_df.to_csv("states.csv")
city_df.to_csv("cities.csv")
```


# Mapping of the data 
I also used these files to create some maps to visualize the data.


![png](/images/project4/Mean.png)

![png](/images/project4/Size_count.png)

![png](/images/project4/Size.png)
