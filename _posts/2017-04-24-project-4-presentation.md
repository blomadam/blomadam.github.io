---
layout: post
title: Web Scraping and Analysis Presentation
date: 2017-04-24
published: false
categories: projects
tags:
---

# Overview

Our latest assignment involved collecting data on Data Science job listings from Indeed.com and using those with salaries to classify others into "High" and "low" categories.

I split this task into several smaller pieces.
- Background research
  + I scraped the Indeed salary summary page for Data Scientist to capture:
    - State statistics (for visualizations)
    - City survey respondents and  salary info (min, max, and avg)
    - List of cities with largest markets and highest average salaries
  + Job listing scraping
    - Pull statistics for each job posting (search term, location, job title, company name, summary description, date posted, company rating, number of company reviews, and advertised salary)
    - Use list from previous step (82 cities)
    - Filter duplicates and clean data
    - Store a cache for later use
  + Analysis
    - Looked to reduce dimensionality by grouping cities into tiers
    - Looked to understand salary distributions
    - Fitted a logistic regression classifier (with K-Folds validation)
  + Presentation
    - I paired with a colleague to create a presentation to show the class our work
    - [Our Team Presentation](https://docs.google.com/presentation/d/1hUeL9EFjfAPdO_G0NjnfBzZukZ_1X-AEMK9XSr6B04A/)
