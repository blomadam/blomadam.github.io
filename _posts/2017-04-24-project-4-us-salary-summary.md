

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
```

## Collect salary summary info from Indeed statistics pages
Indeed has published Data Scientist salary information on [their site](https://www.indeed.com/salaries/Data-Scientist-Salaries).  The publish data averaged over the entire country, for many (~35) individual states, and for several cities in each of those states.  I am looping over each page in their location list to collect information on the salary average and range and an indication of the size of the market in each location.


```python
URL = "https://www.indeed.com/salaries/Data-Scientist-Salaries"
```


```python
res = requests.get(URL)
page_source = BeautifulSoup(res.content,"lxml")
```


```python
#print page_source.prettify()
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
city_URLs[:13]
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
     ('Delaware', u'38', u'$37,000', u'$217,000', u'$108,691\xa0per year'),
     ('District of Columbia',
      u'721',
      u'$38,000',
      u'$244,000',
      u'$119,036\xa0per year'),
     ('Florida', u'347', u'$30,000', u'$215,000', u'$101,818\xa0per year'),
     ('Georgia', u'605', u'$40,000', u'$211,000', u'$109,008\xa0per year'),
     ('Idaho', u'21', u'$30,000', u'$213,000', u'$101,036\xa0per year'),
     ('Illinois', u'1,654', u'$52,000', u'$236,000', u'$127,588\xa0per year'),
     ('Indiana', u'99', u'$27,000', u'$210,000', u'$96,778\xa0per year'),
     ('Iowa', u'20', u'$31,000', u'$172,000', u'$87,324\xa0per year'),
     ('Kansas', u'37', u'$49,000', u'$283,000', u'$142,284\xa0per year'),
     ('Kentucky', u'14', u'$33,000', u'$161,000', u'$85,755\xa0per year'),
     ('Maine', u'6', u'$42,000', u'$151,000', u'$88,840\xa0per year'),
     ('Maryland', u'478', u'$29,000', u'$221,000', u'$102,745\xa0per year'),
     ('Massachusetts', u'5,636', u'$58,000', u'$218,000', u'$125,402\xa0per year'),
     ('Michigan', u'231', u'$39,000', u'$190,000', u'$100,935\xa0per year'),
     ('Minnesota', u'65', u'$36,000', u'$181,000', u'$94,514\xa0per year'),
     ('Missouri', u'281', u'$39,000', u'$192,000', u'$101,471\xa0per year'),
     ('Nebraska', u'18', u'$35,000', u'$207,000', u'$103,791\xa0per year'),
     ('Nevada', u'51', u'$24,000', u'$250,000', u'$107,953\xa0per year'),
     ('New Jersey', u'693', u'$36,000', u'$252,000', u'$120,158\xa0per year'),
     ('New York State',
      u'4,811',
      u'$44,000',
      u'$283,000',
      u'$138,357\xa0per year'),
     ('North Carolina', u'429', u'$28,000', u'$271,000', u'$118,285\xa0per year'),
     ('Ohio', u'278', u'$35,000', u'$193,000', u'$98,514\xa0per year'),
     ('Oklahoma', u'19', u'$30,000', u'$254,000', u'$114,880\xa0per year'),
     ('Oregon', u'380', u'$68,000', u'$234,000', u'$139,315\xa0per year'),
     ('Pennsylvania', u'431', u'$41,000', u'$221,000', u'$113,632\xa0per year'),
     ('South Carolina', u'37', u'$52,000', u'$179,000', u'$106,236\xa0per year'),
     ('Tennessee', u'102', u'$34,000', u'$200,000', u'$100,626\xa0per year'),
     ('Texas', u'1,215', u'$39,000', u'$246,000', u'$120,708\xa0per year'),
     ('Utah', u'561', u'$47,000', u'$207,000', u'$113,416\xa0per year'),
     ('Virginia', u'866', u'$47,000', u'$221,000', u'$118,200\xa0per year'),
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
     ('Costa Mesa', 'CA', u'5', u'$60,000', u'$288,000', u'$153,656\xa0per year'),
     ('Culver City',
      'CA',
      u'381',
      u'$49,000',
      u'$294,000',
      u'$146,538\xa0per year'),
     ('Cupertino', 'CA', u'22', u'$77,000', u'$234,000', u'$154,225\xa0per year'),
     ('Danville', 'CA', u'6', u'$62,000', u'$245,000', u'$138,932\xa0per year'),
     ('Davis', 'CA', u'10', u'$2,300', u'$14,000', u'$6,799\xa0per month'),
     ('El Segundo', 'CA', u'40', u'$45,000', u'$242,000', u'$123,935\xa0per year'),
     ('Foster City',
      'CA',
      u'27',
      u'$54,000',
      u'$264,000',
      u'$139,994\xa0per year'),
     ('Fremont', 'CA', u'10', u'$34,000', u'$247,000', u'$116,467\xa0per year'),
     ('Hollywood', 'CA', u'11', u'$38,000', u'$296,000', u'$136,597\xa0per year'),
     ('Irvine', 'CA', u'294', u'$47,000', u'$196,000', u'$109,123\xa0per year'),
     ('La Jolla', 'CA', u'9', u'$52,000', u'$194,000', u'$112,375\xa0per year'),
     ('Lake Forest', 'CA', u'7', u'$62,000', u'$204,000', u'$123,463\xa0per year'),
     ('Larkspur', 'CA', u'15', u'$70,000', u'$210,000', u'$138,941\xa0per year'),
     ('Los Angeles',
      'CA',
      u'610',
      u'$49,000',
      u'$254,000',
      u'$132,062\xa0per year'),
     ('Los Gatos', 'CA', u'105', u'$78,000', u'$359,000', u'$193,215\xa0per year'),
     ('Marina del Rey',
      'CA',
      u'119',
      u'$85,000',
      u'$268,000',
      u'$167,764\xa0per year'),
     ('Menlo Park',
      'CA',
      u'188',
      u'$67,000',
      u'$218,000',
      u'$133,125\xa0per year'),
     ('Milpitas', 'CA', u'10', u'$63,000', u'$200,000', u'$125,715\xa0per year'),
     ('Modesto', 'CA', u'7', u'$57,000', u'$173,000', u'$113,790\xa0per year'),
     ('Mountain View',
      'CA',
      u'682',
      u'$69,000',
      u'$250,000',
      u'$145,966\xa0per year'),
     ('Newport Beach',
      'CA',
      u'17',
      u'$57,000',
      u'$245,000',
      u'$134,947\xa0per year'),
     ('Oakland', 'CA', u'11', u'$14,000', u'$151,000', u'$50,619\xa0per year'),
     ('Palo Alto',
      'CA',
      u'1,127',
      u'$61,000',
      u'$264,000',
      u'$145,239\xa0per year'),
     ('Pasadena', 'CA', u'176', u'$60,000', u'$242,000', u'$136,061\xa0per year'),
     ('Redwood City',
      'CA',
      u'722',
      u'$67,000',
      u'$279,000',
      u'$155,545\xa0per year'),
     ('San Bruno', 'CA', u'16', u'$73,000', u'$220,000', u'$145,759\xa0per year'),
     ('San Carlos', 'CA', u'67', u'$51,000', u'$298,000', u'$149,779\xa0per year'),
     ('San Diego', 'CA', u'324', u'$52,000', u'$259,000', u'$136,234\xa0per year'),
     ('San Francisco',
      'CA',
      u'4,488',
      u'$50,000',
      u'$274,000',
      u'$140,286\xa0per year'),
     ('San Francisco Bay Area',
      'CA',
      u'30',
      u'$73,000',
      u'$272,000',
      u'$156,986\xa0per year'),
     ('San Jose', 'CA', u'819', u'$65,000', u'$256,000', u'$144,811\xa0per year'),
     ('San Mateo', 'CA', u'567', u'$66,000', u'$245,000', u'$141,909\xa0per year'),
     ('San Ramon', 'CA', u'103', u'$73,000', u'$233,000', u'$145,053\xa0per year'),
     ('Santa Ana', 'CA', u'7', u'$32,000', u'$272,000', u'$122,806\xa0per year'),
     ('Santa Clara',
      'CA',
      u'163',
      u'$56,000',
      u'$264,000',
      u'$141,601\xa0per year'),
     ('Santa Monica',
      'CA',
      u'335',
      u'$41,000',
      u'$314,000',
      u'$146,194\xa0per year'),
     ('Sausalito', 'CA', u'45', u'$37,000', u'$214,000', u'$107,593\xa0per year'),
     ('Scotts Valley',
      'CA',
      u'6',
      u'$59,000',
      u'$180,000',
      u'$117,761\xa0per year'),
     ('Sherman Oaks',
      'CA',
      u'190',
      u'$51,000',
      u'$251,000',
      u'$132,655\xa0per year'),
     ('South San Francisco',
      'CA',
      u'15',
      u'$57,000',
      u'$328,000',
      u'$165,416\xa0per year'),
     ('Sunnyvale', 'CA', u'622', u'$64,000', u'$264,000', u'$147,720\xa0per year'),
     ('Torrance', 'CA', u'68', u'$73,000', u'$237,000', u'$144,222\xa0per year'),
     ('Venice', 'CA', u'137', u'$48,000', u'$265,000', u'$135,545\xa0per year'),
     ('Walnut Creek', 'CA', u'7', u'$7.50', u'$22.50', u'$15.00\xa0per hour'),
     ('West Hollywood',
      'CA',
      u'49',
      u'$44,000',
      u'$343,000',
      u'$158,234\xa0per year'),
     ('Aurora', 'CO', u'7', u'$54,000', u'$169,000', u'$107,035\xa0per year'),
     ('Boulder', 'CO', u'98', u'$53,000', u'$230,000', u'$126,218\xa0per year'),
     ('Colorado Springs',
      'CO',
      u'24',
      u'$51,000',
      u'$154,000',
      u'$101,138\xa0per year'),
     ('Denver', 'CO', u'290', u'$47,000', u'$184,000', u'$104,730\xa0per year'),
     ('Englewood', 'CO', u'13', u'$50,000', u'$228,000', u'$123,505\xa0per year'),
     ('Fort Collins',
      'CO',
      u'14',
      u'$39,000',
      u'$122,000',
      u'$77,791\xa0per year'),
     ('Westminster', 'CO', u'6', u'$21,000', u'$209,000', u'$90,641\xa0per year'),
     ('Hartford', 'CT', u'29', u'$44,000', u'$175,000', u'$99,355\xa0per year'),
     ('Norwalk', 'CT', u'38', u'$57,000', u'$174,000', u'$114,342\xa0per year'),
     ('Stamford', 'CT', u'23', u'$33,000', u'$243,000', u'$114,199\xa0per year'),
     ('Newark', 'DE', u'14', u'$28,000', u'$238,000', u'$107,533\xa0per year'),
     ('Wilmington', 'DE', u'24', u'$48,000', u'$195,000', u'$109,572\xa0per year'),
     ('Washington',
      'DC',
      u'721',
      u'$38,000',
      u'$244,000',
      u'$119,036\xa0per year'),
     ('Boca Raton', 'FL', u'77', u'$26.80', u'$98.30', u'$57.04\xa0per hour'),
     ('Deerfield Beach', 'FL', u'13', u'$36.60', u'$200', u'$73.54\xa0per hour'),
     ('Fort Lauderdale',
      'FL',
      u'32',
      u'$21,000',
      u'$267,000',
      u'$108,783\xa0per year'),
     ('Jacksonville',
      'FL',
      u'30',
      u'$56,000',
      u'$188,000',
      u'$112,271\xa0per year'),
     ('Miami', 'FL', u'27', u'$34,000', u'$222,000', u'$107,635\xa0per year'),
     ('Oldsmar', 'FL', u'5', u'$32,000', u'$98,000', u'$64,244\xa0per year'),
     ('Orlando', 'FL', u'42', u'$40,000', u'$180,000', u'$97,739\xa0per year'),
     ('Saint Petersburg',
      'FL',
      u'12',
      u'$34,000',
      u'$201,000',
      u'$100,180\xa0per year'),
     ('St Petersburg Beach',
      'FL',
      u'8',
      u'$70,000',
      u'$210,000',
      u'$140,000\xa0per year'),
     ('Tampa', 'FL', u'49', u'$33,000', u'$231,000', u'$109,765\xa0per year'),
     ('Winter Park',
      'FL',
      u'75',
      u'$35,000',
      u'$213,000',
      u'$105,678\xa0per year'),
     ('Alpharetta', 'GA', u'52', u'$41,000', u'$144,000', u'$85,039\xa0per year'),
     ('Atlanta', 'GA', u'436', u'$44,000', u'$222,000', u'$116,282\xa0per year'),
     ('Augusta', 'GA', u'20', u'$37,000', u'$113,000', u'$74,548\xa0per year'),
     ('Duluth', 'GA', u'54', u'$47,000', u'$179,000', u'$102,734\xa0per year'),
     ('Jersey', 'GA', u'7', u'$40,000', u'$122,000', u'$80,673\xa0per year'),
     ('Norcross', 'GA', u'10', u'$40,000', u'$164,000', u'$91,819\xa0per year'),
     ('Tifton', 'GA', u'17', u'$52,000', u'$158,000', u'$104,881\xa0per year'),
     ('Boise', 'ID', u'19', u'$40,000', u'$213,000', u'$109,451\xa0per year'),
     ('Champaign', 'IL', u'8', u'$40,000', u'$223,000', u'$113,041\xa0per year'),
     ('Chicago', 'IL', u'1,349', u'$54,000', u'$229,000', u'$126,485\xa0per year'),
     ('Deerfield', 'IL', u'14', u'$63,000', u'$190,000', u'$125,899\xa0per year'),
     ('Des Plaines',
      'IL',
      u'27',
      u'$64,000',
      u'$211,000',
      u'$126,808\xa0per year'),
     ('Downers Grove',
      'IL',
      u'8',
      u'$77,000',
      u'$238,000',
      u'$153,647\xa0per year'),
     ('Evanston', 'IL', u'32', u'$58,000', u'$233,000', u'$130,919\xa0per year'),
     ('Hoffman Estates',
      'IL',
      u'13',
      u'$64,000',
      u'$254,000',
      u'$143,521\xa0per year'),
     ('Lisle', 'IL', u'13', u'$7.25', u'$53.95', u'$19.04\xa0per hour'),
     ('Naperville', 'IL', u'22', u'$43,000', u'$195,000', u'$106,093\xa0per year'),
     ('Rosemont', 'IL', u'24', u'$46,000', u'$139,000', u'$91,695\xa0per year'),
     ('Schaumburg',
      'IL',
      u'119',
      u'$64,000',
      u'$288,000',
      u'$156,593\xa0per year'),
     ('Carmel', 'IN', u'7', u'$59,000', u'$178,000', u'$118,460\xa0per year'),
     ('Fishers', 'IN', u'22', u'$22,000', u'$147,000', u'$71,151\xa0per year'),
     ('Indianapolis',
      'IN',
      u'53',
      u'$36,000',
      u'$232,000',
      u'$113,542\xa0per year'),
     ('Lafayette', 'IN', u'5', u'$36,000', u'$123,000', u'$73,544\xa0per year'),
     ('Des Moines', 'IA', u'9', u'$60.00', u'$200', u'$85.00\xa0per day'),
     ('Overland Park',
      'KS',
      u'29',
      u'$59,000',
      u'$280,000',
      u'$149,494\xa0per year'),
     ('Louisville', 'KY', u'8', u'$46,000', u'$160,000', u'$94,336\xa0per year'),
     ('New Gloucester',
      'ME',
      u'5',
      u'$48,000',
      u'$147,000',
      u'$96,708\xa0per year'),
     ('Annapolis', 'MD', u'10', u'$34,000', u'$248,000', u'$116,562\xa0per year'),
     ('Baltimore', 'MD', u'67', u'$34,000', u'$269,000', u'$124,307\xa0per year'),
     ('Bethesda', 'MD', u'45', u'$50,000', u'$248,000', u'$130,589\xa0per year'),
     ('Chevy Chase',
      'MD',
      u'15',
      u'$41,000',
      u'$314,000',
      u'$146,051\xa0per year'),
     ('Columbia', 'MD', u'28', u'$56,000', u'$210,000', u'$121,360\xa0per year'),
     ('Fort George G Meade',
      'MD',
      u'142',
      u'$34,000',
      u'$154,000',
      u'$83,240\xa0per year'),
     ('Fort Meade', 'MD', u'86', u'$33,000', u'$169,000', u'$88,457\xa0per year'),
     ('Greenbelt', 'MD', u'7', u'$37,000', u'$117,000', u'$73,928\xa0per year'),
     ('Hughesville', 'MD', u'6', u'$42,000', u'$128,000', u'$84,484\xa0per year'),
     ('Middletown', 'MD', u'21', u'$52,000', u'$244,000', u'$131,309\xa0per year'),
     ('Rockville', 'MD', u'16', u'$46,000', u'$255,000', u'$130,080\xa0per year'),
     ('Amherst', 'MA', u'14', u'$30,000', u'$108,000', u'$63,683\xa0per year'),
     ('Bedford', 'MA', u'46', u'$72,000', u'$218,000', u'$143,155\xa0per year'),
     ('Beverly', 'MA', u'5', u'$55,000', u'$175,000', u'$108,226\xa0per year'),
     ('Boston', 'MA', u'1,955', u'$53,000', u'$211,000', u'$119,223\xa0per year'),
     ('Brockton', 'MA', u'9', u'$13.45', u'$40.45', u'$26.90\xa0per hour'),
     ('Cambridge',
      'MA',
      u'2,093',
      u'$62,000',
      u'$228,000',
      u'$132,209\xa0per year'),
     ('Charlestown', 'MA', u'9', u'$60,000', u'$181,000', u'$119,882\xa0per year'),
     ('Framingham', 'MA', u'28', u'$52,000', u'$219,000', u'$121,604\xa0per year'),
     ('Lexington', 'MA', u'37', u'$59,000', u'$193,000', u'$117,055\xa0per year'),
     ('Lowell', 'MA', u'6', u'$46,000', u'$140,000', u'$92,581\xa0per year'),
     ('Marlborough',
      'MA',
      u'721',
      u'$65,000',
      u'$198,000',
      u'$129,984\xa0per year'),
     ('Natick', 'MA', u'378', u'$57,000', u'$224,000', u'$126,951\xa0per year'),
     ('Needham', 'MA', u'42', u'$60,000', u'$212,000', u'$124,654\xa0per year'),
     ('Newton', 'MA', u'145', u'$59,000', u'$192,000', u'$117,769\xa0per year'),
     ('Somerville', 'MA', u'41', u'$40,000', u'$174,000', u'$95,755\xa0per year'),
     ('Watertown', 'MA', u'8', u'$71,000', u'$222,000', u'$141,796\xa0per year'),
     ('Wellesley', 'MA', u'17', u'$65,000', u'$198,000', u'$130,222\xa0per year'),
     ('Worcester', 'MA', u'14', u'$45,000', u'$236,000', u'$122,362\xa0per year'),
     ('Ann Arbor', 'MI', u'33', u'$39,000', u'$119,000', u'$77,839\xa0per year'),
     ('Auburn Hills', 'MI', u'7', u'$27.10', u'$81.45', u'$54.26\xa0per hour'),
     ('Dearborn', 'MI', u'23', u'$20.75', u'$92.85', u'$50.48\xa0per hour'),
     ('Detroit', 'MI', u'124', u'$47,000', u'$200,000', u'$110,373\xa0per year'),
     ('Livonia', 'MI', u'9', u'$45,000', u'$189,000', u'$104,632\xa0per year'),
     ('Troy', 'MI', u'18', u'$55,000', u'$166,000', u'$109,823\xa0per year'),
     ('Eden Prairie',
      'MN',
      u'14',
      u'$57,000',
      u'$181,000',
      u'$113,021\xa0per year'),
     ('Minneapolis', 'MN', u'46', u'$32,000', u'$178,000', u'$90,823\xa0per year'),
     ('Chesterfield',
      'MO',
      u'24',
      u'$28,000',
      u'$188,000',
      u'$90,503\xa0per year'),
     ('Kansas City', 'MO', u'59', u'$31,000', u'$177,000', u'$89,472\xa0per year'),
     ('St. Louis', 'MO', u'183', u'$46,000', u'$195,000', u'$107,821\xa0per year'),
     ('Lincoln', 'NE', u'14', u'$49,000', u'$211,000', u'$115,896\xa0per year'),
     ('Henderson', 'NV', u'20', u'$60,000', u'$289,000', u'$154,098\xa0per year'),
     ('Las Vegas', 'NV', u'26', u'$27,000', u'$160,000', u'$79,592\xa0per year'),
     ('Bridgewater',
      'NJ',
      u'15',
      u'$43,000',
      u'$213,000',
      u'$111,938\xa0per year'),
     ('Edison', 'NJ', u'13', u'$31.05', u'$93.25', u'$61.98\xa0per hour'),
     ('Fort Lee', 'NJ', u'7', u'$37,000', u'$238,000', u'$115,781\xa0per year'),
     ('Hawthorne', 'NJ', u'16', u'$72,000', u'$217,000', u'$143,689\xa0per year'),
     ('Hazlet', 'NJ', u'12', u'$14,000', u'$130,000', u'$55,809\xa0per year'),
     ('Iselin', 'NJ', u'7', u'$30.00', u'$90.00', u'$60.00\xa0per hour'),
     ('Jersey City',
      'NJ',
      u'255',
      u'$57,000',
      u'$180,000',
      u'$112,275\xa0per year'),
     ('Mahwah', 'NJ', u'38', u'$69,000', u'$221,000', u'$135,769\xa0per year'),
     ('Newark', 'NJ', u'11', u'$55,000', u'$206,000', u'$118,446\xa0per year'),
     ('Piscataway', 'NJ', u'80', u'$42,000', u'$129,000', u'$82,981\xa0per year'),
     ('Plainsboro', 'NJ', u'14', u'$82.40', u'$300', u'$165\xa0per day'),
     ('Princeton', 'NJ', u'57', u'$55,000', u'$245,000', u'$133,136\xa0per year'),
     ('Saddle Brook', 'NJ', u'8', u'$31.25', u'$93.75', u'$62.45\xa0per hour'),
     ('Somerville', 'NJ', u'8', u'$50,000', u'$152,000', u'$100,404\xa0per year'),
     ('South Hackensack',
      'NJ',
      u'6',
      u'$67,000',
      u'$322,000',
      u'$171,467\xa0per year'),
     ('South Plainfield',
      'NJ',
      u'81',
      u'$93,000',
      u'$347,000',
      u'$200,233\xa0per year'),
     ('Union', 'NJ', u'21', u'$68,000', u'$235,000', u'$139,275\xa0per year'),
     ('Brentwood', 'NY', u'7', u'$40,000', u'$120,000', u'$80,000\xa0per year'),
     ('Brooklyn', 'NY', u'48', u'$64,000', u'$229,000', u'$134,089\xa0per year'),
     ('Buffalo', 'NY', u'12', u'$22,000', u'$159,000', u'$74,968\xa0per year'),
     ('Levittown', 'NY', u'85', u'$60,000', u'$181,000', u'$119,704\xa0per year'),
     ('Manhattan', 'NY', u'169', u'$76,000', u'$309,000', u'$173,056\xa0per year'),
     ('New York',
      'NY',
      u'4,306',
      u'$45,000',
      u'$284,000',
      u'$139,598\xa0per year'),
     ('Philadelphia',
      'NY',
      u'20',
      u'$80,000',
      u'$243,000',
      u'$160,839\xa0per year'),
     ('Port Washington',
      'NY',
      u'16',
      u'$57,000',
      u'$300,000',
      u'$154,908\xa0per year'),
     ('Troy', 'NY', u'28', u'$67,000', u'$215,000', u'$133,546\xa0per year'),
     ('Cary', 'NC', u'9', u'$60.00', u'$200', u'$70.00\xa0per day'),
     ('Charlotte', 'NC', u'52', u'$41,000', u'$209,000', u'$109,047\xa0per year'),
     ('Durham', 'NC', u'73', u'$52,000', u'$174,000', u'$104,617\xa0per year'),
     ('Huntersville', 'NC', u'8', u'$32,000', u'$98,000', u'$64,772\xa0per year'),
     ('Morrisville',
      'NC',
      u'15',
      u'$53,000',
      u'$229,000',
      u'$125,892\xa0per year'),
     ('Raleigh', 'NC', u'248', u'$29,000', u'$304,000', u'$129,885\xa0per year'),
     ('Research Triangle Park',
      'NC',
      u'9',
      u'$61,000',
      u'$207,000',
      u'$123,657\xa0per year'),
     ('Akron', 'OH', u'71', u'$52,000', u'$213,000', u'$119,298\xa0per year'),
     ('Broadview Heights',
      'OH',
      u'6',
      u'$60,000',
      u'$180,000',
      u'$120,000\xa0per year'),
     ('Cleveland', 'OH', u'95', u'$43,000', u'$144,000', u'$86,520\xa0per year'),
     ('Columbus', 'OH', u'50', u'$43,000', u'$197,000', u'$106,417\xa0per year'),
     ('Mentor', 'OH', u'8', u'$37,000', u'$113,000', u'$74,505\xa0per year'),
     ('Oklahoma City', 'OK', u'9', u'$14.70', u'$79.55', u'$40.77\xa0per hour'),
     ('Roland', 'OK', u'5', u'$95,000', u'$289,000', u'$187,624\xa0per year'),
     ('Bend', 'OR', u'8', u'$72,000', u'$218,000', u'$144,914\xa0per year'),
     ('Corvallis', 'OR', u'5', u'$67,000', u'$216,000', u'$132,743\xa0per year'),
     ('Hillsboro', 'OR', u'7', u'$53,000', u'$188,000', u'$110,151\xa0per year'),
     ('Portland', 'OR', u'351', u'$71,000', u'$234,000', u'$141,103\xa0per year'),
     ('King of Prussia',
      'PA',
      u'22',
      u'$52,000',
      u'$157,000',
      u'$103,027\xa0per year'),
     ('Malvern', 'PA', u'18', u'$53,000', u'$168,000', u'$105,190\xa0per year'),
     ('Philadelphia',
      'PA',
      u'158',
      u'$36,000',
      u'$226,000',
      u'$110,895\xa0per year'),
     ('Pittsburgh',
      'PA',
      u'110',
      u'$56,000',
      u'$225,000',
      u'$126,759\xa0per year'),
     ('Lexington', 'SC', u'23', u'$58,000', u'$175,000', u'$115,980\xa0per year'),
     ('Nashville', 'TN', u'82', u'$38,000', u'$194,000', u'$101,769\xa0per year'),
     ('Addison', 'TX', u'21', u'$27,000', u'$181,000', u'$87,587\xa0per year'),
     ('Austin', 'TX', u'437', u'$35,000', u'$277,000', u'$127,416\xa0per year'),
     ('Dallas', 'TX', u'310', u'$51,000', u'$226,000', u'$122,996\xa0per year'),
     ('Houston', 'TX', u'200', u'$52,000', u'$221,000', u'$122,365\xa0per year'),
     ('Irving', 'TX', u'74', u'$65,000', u'$201,000', u'$128,780\xa0per year'),
     ('Midland', 'TX', u'5', u'$78,000', u'$237,000', u'$156,537\xa0per year'),
     ('North Richland Hills',
      'TX',
      u'11',
      u'$46,000',
      u'$157,000',
      u'$93,572\xa0per year'),
     ('Plano', 'TX', u'22', u'$53,000', u'$175,000', u'$105,299\xa0per year'),
     ('San Antonio',
      'TX',
      u'74',
      u'$47,000',
      u'$172,000',
      u'$100,274\xa0per year'),
     ('Spring', 'TX', u'7', u'$55,000', u'$175,000', u'$108,226\xa0per year'),
     ('Sugar Land', 'TX', u'11', u'$52,000', u'$246,000', u'$131,802\xa0per year'),
     ('Draper', 'UT', u'9', u'$53,000', u'$174,000', u'$104,953\xa0per year'),
     ('Lehi', 'UT', u'10', u'$61,000', u'$186,000', u'$122,243\xa0per year'),
     ('Provo', 'UT', u'6', u'$34,000', u'$112,000', u'$67,912\xa0per year'),
     ('Salt Lake City',
      'UT',
      u'517',
      u'$48,000',
      u'$207,000',
      u'$113,832\xa0per year'),
     ('West Jordan',
      'UT',
      u'15',
      u'$54,000',
      u'$200,000',
      u'$115,631\xa0per year'),
     ('Alexandria', 'VA', u'39', u'$56,000', u'$224,000', u'$126,363\xa0per year'),
     ('Arlington', 'VA', u'101', u'$48,000', u'$236,000', u'$124,555\xa0per year'),
     ('Chantilly', 'VA', u'34', u'$44,000', u'$183,000', u'$101,991\xa0per year'),
     ('Charlottesville',
      'VA',
      u'24',
      u'$45,000',
      u'$177,000',
      u'$100,265\xa0per year'),
     ('Elkton', 'VA', u'5', u'$57,000', u'$173,000', u'$114,034\xa0per year'),
     ('Falls Church',
      'VA',
      u'9',
      u'$55,000',
      u'$187,000',
      u'$111,809\xa0per year'),
     ('Leesburg', 'VA', u'12', u'$50,000', u'$150,000', u'$99,504\xa0per year'),
     ('McLean', 'VA', u'487', u'$47,000', u'$221,000', u'$118,260\xa0per year'),
     ('Reston', 'VA', u'30', u'$68,000', u'$217,000', u'$134,811\xa0per year'),
     ('Richmond', 'VA', u'43', u'$36,000', u'$222,000', u'$110,060\xa0per year'),
     ('Springfield',
      'VA',
      u'36',
      u'$66,000',
      u'$218,000',
      u'$131,497\xa0per year'),
     ('Bellevue', 'WA', u'212', u'$62,000', u'$201,000', u'$122,193\xa0per year'),
     ('Redmond', 'WA', u'345', u'$54,000', u'$251,000', u'$134,885\xa0per year'),
     ('Renton', 'WA', u'22', u'$70,000', u'$213,000', u'$141,185\xa0per year'),
     ('Seattle', 'WA', u'609', u'$50,000', u'$252,000', u'$131,987\xa0per year'),
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
<table border="1" class="dataframe">
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
<table border="1" class="dataframe">
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
<table border="1" class="dataframe">
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
     'Washington%2C+DC',
     'Marlborough%2C+MA',
     'Mountain+View%2C+CA',
     'Sunnyvale%2C+CA',
     'Los+Angeles%2C+CA',
     'Seattle%2C+WA',
     'San+Mateo%2C+CA',
     'Berkeley%2C+CA',
     'Salt+Lake+City%2C+UT',
     'McLean%2C+VA',
     'Austin%2C+TX',
     'Atlanta%2C+GA',
     'Culver+City%2C+CA',
     'Natick%2C+MA',
     'Portland%2C+OR',
     'Redmond%2C+WA',
     'Santa+Monica%2C+CA',
     'San+Diego%2C+CA',
     'Dallas%2C+TX',
     'Irvine%2C+CA',
     'Denver%2C+CO',
     'Jersey+City%2C+NJ',
     'Raleigh%2C+NC',
     'Bellevue%2C+WA',
     'Houston%2C+TX',
     'Sherman+Oaks%2C+CA',
     'Menlo+Park%2C+CA',
     'St.+Louis%2C+MO',
     'Pasadena%2C+CA',
     'Manhattan%2C+NY',
     'Santa+Clara%2C+CA',
     'Philadelphia%2C+PA',
     'Carlsbad%2C+CA',
     'Newton%2C+MA',
     'Fort+George+G+Meade%2C+MD',
     'Venice%2C+CA',
     'Detroit%2C+MI',
     'Marina+del+Rey%2C+CA',
     'Schaumburg%2C+IL',
     'Pittsburgh%2C+PA',
     'Los+Gatos%2C+CA',
     'San+Ramon%2C+CA',
     'South+Plainfield%2C+NJ',
     'Los+Gatos%2C+CA',
     'Roland%2C+OK',
     'Manhattan%2C+NY',
     'South+Hackensack%2C+NJ',
     'Marina+del+Rey%2C+CA',
     'South+San+Francisco%2C+CA',
     'Philadelphia%2C+NY',
     'West+Hollywood%2C+CA',
     'San+Francisco+Bay+Area%2C+CA',
     'Schaumburg%2C+IL',
     'Midland%2C+TX',
     'Redwood+City%2C+CA',
     'Port+Washington%2C+NY',
     'Belmont%2C+CA',
     'Phoenix%2C+AZ',
     'Cupertino%2C+CA',
     'Henderson%2C+NV',
     'Costa+Mesa%2C+CA',
     'Downers+Grove%2C+IL',
     'San+Carlos%2C+CA',
     'Overland+Park%2C+KS',
     'Sunnyvale%2C+CA',
     'Deerfield+Beach%2C+FL',
     'Culver+City%2C+CA',
     'Santa+Monica%2C+CA',
     'Berkeley%2C+CA',
     'Chevy+Chase%2C+MD',
     'Mountain+View%2C+CA',
     'San+Bruno%2C+CA',
     'Palo+Alto%2C+CA',
     'San+Ramon%2C+CA',
     'Bend%2C+OR',
     'San+Jose%2C+CA',
     'Torrance%2C+CA',
     'Campbell%2C+CA',
     'Hawthorne%2C+NJ',
     'Hoffman+Estates%2C+IL',
     'Bedford%2C+MA',
     'San+Mateo%2C+CA',
     'Watertown%2C+MA',
     'Santa+Clara%2C+CA',
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


```python

```
