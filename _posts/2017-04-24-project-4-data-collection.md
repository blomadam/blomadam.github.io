---
layout: post
title: US Salary Summary
date: 2017-04-23
published: false
categories: projects
image: /images/project4/Size.png
---

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
```

## Functions to parse results


```python
def extract_location_from_job_result(result):
    a = result.find('span',class_="location")
    return None if not a else a.text.strip()

def extract_company_from_job_result(result):
    a = result.find('span',class_="company")
    return None if not a else a.text.strip()

def extract_jobtitle_from_job_result(result):
    a = result.find('a', {"data-tn-element":"jobTitle"})
    return None if not a else a.text.strip()
    
def extract_salary_from_job_result(result):
    a = result.find('nobr', text=True)
    return None if not a else a.text.strip()

def extract_date_from_job_result(result):
    a = result.find('span', class_="date")
    return None if not a else a.text.strip()

def extract_summary_from_job_result(result):
    a = result.find('span', class_="summary")
    return None if not a else a.text.strip()

def extract_rating_from_job_result(result):
    a = result.find('span',class_="rating")
    return None if not a else float(a.attrs['style'].split(':')[1][:-2])

def extract_reviews_from_job_result(result):
    a = result.find('span',class_="slNoUnderline")
    return None if not a else int(a.text.strip().replace(",","").split()[0])
```

## Loop to collect data

I collected summary data for the country, and selected the 50 cities with highest average salary and 50 more with the largest number of respondents to the salary survey.  This resulted in roughly 80 cities across the nation for my search pool.  From this data, I deleted any duplicate rows.


```python
url_template = "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l={}&start={}"
max_results_per_city = 300

results = []

cities = ['San+Francisco%2C+CA', 'New+York%2C+NY', 'Cambridge%2C+MA', 'Boston%2C+MA', \
 'Chicago%2C+IL', 'Palo+Alto%2C+CA', 'San+Jose%2C+CA', 'Redwood+City%2C+CA', \
 'Washington%2C+DC', 'Marlborough%2C+MA', 'Mountain+View%2C+CA', 'Sunnyvale%2C+CA', \
 'Los+Angeles%2C+CA', 'Seattle%2C+WA', 'San+Mateo%2C+CA', 'Berkeley%2C+CA', \
 'Salt+Lake+City%2C+UT', 'McLean%2C+VA', 'Austin%2C+TX', 'Atlanta%2C+GA', \
 'Culver+City%2C+CA', 'Natick%2C+MA', 'Portland%2C+OR', 'Redmond%2C+WA', \
 'Santa+Monica%2C+CA', 'San+Diego%2C+CA', 'Dallas%2C+TX', 'Irvine%2C+CA', \
 'Denver%2C+CO', 'Jersey+City%2C+NJ', 'Raleigh%2C+NC', 'Bellevue%2C+WA', \
 'Houston%2C+TX', 'Sherman+Oaks%2C+CA', 'Menlo+Park%2C+CA', 'St.+Louis%2C+MO', \
 'Pasadena%2C+CA', 'Manhattan%2C+NY', 'Santa+Clara%2C+CA', 'Philadelphia%2C+PA', \
 'Carlsbad%2C+CA', 'Newton%2C+MA', 'Fort+George+G+Meade%2C+MD', 'Venice%2C+CA', \
 'Detroit%2C+MI', 'Marina+del+Rey%2C+CA', 'Schaumburg%2C+IL', 'Pittsburgh%2C+PA', \
 'Los+Gatos%2C+CA', 'San+Ramon%2C+CA', 'South+Plainfield%2C+NJ', 'Roland%2C+OK', \
 'South+Hackensack%2C+NJ', 'South+San+Francisco%2C+CA', 'Philadelphia%2C+NY', \
 'West+Hollywood%2C+CA', 'San+Francisco+Bay+Area%2C+CA', 'Midland%2C+TX', \
 'Port+Washington%2C+NY', 'Belmont%2C+CA', 'Phoenix%2C+AZ', 'Cupertino%2C+CA', \
 'Henderson%2C+NV', 'Costa+Mesa%2C+CA', 'Downers+Grove%2C+IL', 'San+Carlos%2C+CA', \
 'Overland+Park%2C+KS', 'Deerfield+Beach%2C+FL', 'Chevy+Chase%2C+MD', \
 'San+Bruno%2C+CA', 'Bend%2C+OR', 'Torrance%2C+CA', 'Campbell%2C+CA', 'Hawthorne%2C+NJ', \
 'Hoffman+Estates%2C+IL', 'Bedford%2C+MA', 'Watertown%2C+MA', 'Renton%2C+WA', \
 'St+Petersburg+Beach%2C+FL', 'Kohler%2C+WI', 'Foster+City%2C+CA', 'Union%2C+NJ']

# cities = ['San+Francisco%2C+CA', 'New+York%2C+NY']


for city in cities:
    for start in range(0, max_results_per_city, 10):
        # Grab the results from the request (as above)
        # Append to the full set of results
        res = requests.get(url_template.format(city,start))
#        print city, start, res.status_code
        page_source = BeautifulSoup(res.content,"lxml")
        result_list = page_source.findAll('div',class_="result")
        for i in result_list:
            results.append( ( city,
                             extract_location_from_job_result(i), \
                             extract_company_from_job_result(i), \
                             extract_jobtitle_from_job_result(i), \
                             extract_summary_from_job_result(i), \
                             extract_date_from_job_result(i), \
                             extract_reviews_from_job_result(i), \
                             extract_rating_from_job_result(i), \
                             extract_salary_from_job_result(i) ) \
                          )
```


```python
# print i.prettify()
```

## Check duplicates

Appears only 18.5% of my data are not duplicates.  Additionally, only 6.4% of those have salary information!


```python
len(results)
```




    36137




```python
df = pd.DataFrame(results, columns=["search_city","location","company","jobtitle","summary","date_posted","reviews","ratings","salary"])
df.drop_duplicates(subset=["location","company","jobtitle","summary","salary"], inplace=True)
```


```python
df[df['salary'].notnull()].shape
```




    (426, 9)




```python
df.shape
```




    (6703, 9)



## Fix salary information
I can use the same function from the salary summary workbook to clean my salaries.


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
df["salary"] = df["salary"].map(lambda x: None if x==None else fix_salaries(x))
# df[df['salary'].notnull()].head()
```


```python
df[['location','jobtitle','company','summary']] = df[['location','jobtitle','company','summary']].applymap(lambda x: x.encode('ascii', 'ignore') if x else None)

```


```python
df['search_city'] = df['search_city'].map(lambda x: x.replace("+"," ").replace("%2C",","))
```


```python
df.insert(len(df.columns)-3,'days_ago_posted',df['date_posted'].map(lambda x: None if not x else int(x.replace("+","").split()[0]) if 'days' in x else 0))

```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6703 entries, 0 to 36127
    Data columns (total 10 columns):
    search_city        6703 non-null object
    location           6702 non-null object
    company            6699 non-null object
    jobtitle           6702 non-null object
    summary            6702 non-null object
    date_posted        6496 non-null object
    days_ago_posted    6496 non-null float64
    reviews            4848 non-null float64
    ratings            4848 non-null float64
    salary             426 non-null float64
    dtypes: float64(4), object(6)
    memory usage: 576.0+ KB



```python
df.to_csv("salary_data.csv",index=False)
```
