---
layout: post
title: First Choropleths
---

We worked with some data about SAT scores and test taking rates in every state over the last week.  I was able to use Tableau with some custom mapping options to keep the image looking professional.

I found a file defining polygons for each state at [Tableau Mapping][1].  I was able to follow the directions from a [tutorial by Steve Batt][2] which basically say:

- open the saved map file from [Tableau Mapping][1]
- Make sure the Latitude and Longitude columns are *Measures*
- Drag *Longitude* to the *Columns* shelf and make sure the Aggregation is set to agerage (i.e. it reads *AVG(Longitude) on the pill)
- Drag *Latitude* to the rows shelf and ensure it is an average too




![](../images/project1/Rate.png)
![](../images/project1/Math.png)
![](../images/project1/Verbal.png)



[1]: https://tableaumapping.bi/2013/08/27/usa-states-offset-ak-hi/  "Tableau Maps"

[2]: http://blogs.lib.uconn.edu/outsidetheneatline/2016/05/12/creating-a-custom-polygon-map-for-connecticut-towns-in-tableau/   "Map Tutorial"

