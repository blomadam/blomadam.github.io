---
layout: post
title: Exploring Project Structure and Testing
date: 2022-02-15
published: true
categories: projects
image: /images/github_actions/badge_combo.png
---

I created a repository to explore and document continuous development practices.  Most notably, I want to explore how automated testing using GitHub Actions (or similar) might be useful in my future data science projects.  I'll be automating tests in [pytest][pytest], flake8[flake8], and [mypy][mypy] with [tox][tox], and also connecting to a protected database while ensuring private information is not published.

### Introduction
The impetus for testing these options out came from a video I recently viewed on YouTube by mCoding (see [here][video]).
Of course once I got started there were multiple concepts I wanted to work with, hence the expanding list of resources above.

- Deploying model code as python package with [consistent folder structures][CCDS]
- Methods for reproducing a consistent analysis environment (e.g. [tox][tox])
- Automated testing with [pytest][pytest] and [GitHub Actions][GH-A]
- Writing more modular code that can be unit tested
- Best practices for protecting credentials
- Working with [clinical trial datasets][AACT]


## Setup as a python package
The key to automating code evaluation is to present your code in a way that the testing tools expect.  This is most easily accomplished by adhering to the standards set by software developers for  generic testing and publication.  In my case, organizing my work as a Python package allows access to the necessary tools for automated testing.  The [tutorial on packaging a project][python_projects] does a good job explaining what needs to be done.  In a nutshell, however, I just needed to match some basic folder structures, add some blank template files, and create a setup configuration file with the dependencies for my code.

```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.cfg
├── src/
│   └── example_package/
│       ├── __init__.py
│       └── example.py    <--- My code goes here
└── tests/
    └── test_example.py   <--- Optional unit test code
```

As an added benefit, [Cookiecutter Data Science][CCDS] has published notes on a folder  structure that adheres to these package requirements, but also has additions for Jupyter notebooks, raw/interim/processed datasets, and Makefiles that make typical code and resource flows of the  typical data project easier to manage. Here is a subset showing some of the additions most useful to me.

```
├── Makefile                 
├── data
│   ├── interim       
│   ├── processed     
│   └── raw           
├── notebooks         
├── references        
├── requirements.txt  
├── setup.py          
├── src               
│   ├── __init__.py   
│   ├── data          
│   │   └── make_dataset.py
│   ├── features       
│   │   └── build_features.py
│   ├── models         
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization  
│       └── visualize.py
```

What goes in the various setup/configuration files is best left to [the tutorial][python_projects], or [the video][video] that spurred me to this project.


## Collect data but protect passwords
Since I wanted to explore how testing might work on a data project, I needed a data source.  I decided to use a public, but still protected database in order to review options for accessing credentials in a repository without publishing said credentials.  There are many ways to work thru this issue, [some are more secure than others][safe_keys] and keep the credentials encrypted.  I ultimately decided that level of security is not typically  warranted in my projects, and went with a method using `configparser`, a builtin python package.  In addition to running anywhere python is available, `configparser` uses `.ini` files to store the credentials.  These files are widely used in many programming languages, and are human readable/editable as well.  This makes it easy to publish examples and reference notes for future use without regard to tools available in the final working environment - any computer will be able to work with the plain text files. See [the docs][config_docs] for more details about `configparser` in general, or the next section for how to work with AACT.


#### Setup notes for AACT database
Cloning [my repository][repo] provides most of the files required to run the tests.  The data the tests are based on, however, comes from the Aggregate Analysis of ClinicalTrials.gov (AACT) database.  This database hosts data submitted to [ClinicalTrials.gov]() about proposed medical trials.  Connecting to this account [requires free credentials](https://aact.ctti-clinicaltrials.org/users/sign_up).

To use  your credentials with this repository, copy the `credentials.ini.example` file that should be copied to `credentials.ini`. Then populate `credentials.ini` with your username and password for the AACT database.


## Define some tests
One of the biggest errors I see in sharing machine learning work, is not starting the process from the correct datasets.  Particularly when there is a tight deadline, the temptation to skip changing filenames in the code for a last minute data update or new filter is hard to resist.  For this reason, I decided to add a data file hash check in my unit tests to verify the data is unchanged from the time the unit tests were written.  In case you haven't seen them before, file hashes are a sort of fingerprint for a file.  Hashing the file runs a complicated analysis on the data in the file and returns a seemingly random string.  See some examples from my interim data files:

```
$ md5 *
MD5 (keywords_table.csv.gz) = f8363838960fc63139d58324a43e11c6
MD5 (main_table.csv.gz) = ff7247b0119c3df7a1e593576c206c6f
```

This string (or hash) is unique to that file — and you will get the same result every time you run the hashing algorithm on that file... unless something changed the data in the file.  If someone overwrites the data in the file (say by filling missing values or making a copy that fails halfway) — the hash will completely change to a different seemingly random string.  Hashes can provide powerful protection data corruption, network errors when sharing, or momentary idiocy — if one remembers to check them before starting work.  Hence the power in incorporating this into automated testing.

I used this [tip from stackoverflow][hashing] to run my hashes in python.


## Future work
This project is rapidly consuming much of my free time, and there is a lot more I want to accomplish.  I have currently collected some data, performed preliminary basic EDA on it, and prepared some rudimentary NLP features from each data file.  I still need to join the data into a final dataset and build a model.  Then I will create some tests to give confidence the model has not been corrupted since I saved it. I'd like to refactor my code to modernize some of the feature engineering and make it easier to test and pass new data thru. Finally, I'll setup some automate documentation to help others follow my work.



[tox]: https://github.com/tox-dev/tox
[pytest]: https://docs.pytest.org/en/latest/
[flake8]: https://flake8.pycqa.org/en/latest/
[mypy]: https://mypy.readthedocs.io/en/stable/index.html
[AACT]: https://aact.ctti-clinicaltrials.org/
[video]: https://www.youtube.com/watch?v=DhUpxWjOhME
[repo]: https://github.com/blomadam/autotest_data
[GH-A]: https://docs.github.com/en/actions
[CCDS]: https://drivendata.github.io/cookiecutter-data-science/
[python_projects]: https://packaging.python.org/en/latest/tutorials/packaging-projects/
[config_docs]: https://docs.python.org/3.9/library/configparser.html
[safe_keys]: https://gist.github.com/amelieykw/6116ca8ef7279206382a76fd790c1aa1
[hashing]: https://stackoverflow.com/a/22058673/7862615
