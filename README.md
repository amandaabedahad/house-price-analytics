# Housing analytics - data aquisition and analytics
tbc

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project aims to provide data driven insights of the house market situation today in Gothenburg, to act as an extension of the information already provided on Hemnet's webpage. In addition, the house price prediction serivce is extended to Gothenburg, which currently only is available for Stockholm. 

This project can be divided into three parts: data aqusition, data analytics and app deployment

### Data aquistion
The dataset is created from scratch using the technique called web scraping - this since no datasets containing wanted information are available in Sweden. The data is aquired through web scraping on Hemnet's webpage for sold listings (https://hemnetanalyticsdocker.azurewebsites.net/).
The dataset consists of the following entities: 
* Address
* Housing type
* Region
* City
* Square meter
* Number rooms
* Monthly fee
* Final sold price
* Sold date
* ....

### Data analytics
Statistical analysis and the machine learning algorithm TBC


### App deploymenet
The insights are visualized in a dashboard, deployed on Azure. The machine learning house price prediction serice is integrated.  TBC

## Technologies
Project is created with:
* python: 3.10
* more
	
## Setup

To setup the environment for the project, one can either choose to create an container of the docker file (needs docker installed) or use the requirements file. 

For the second approach, using the requirements file, run the following lines of code.
```
$ pip install -r requirements.txt
```

To scrape data, and store in a database, add a .env file with data base credentials - and ```$ python main.py``` to scrape the current data from Hemnet. 

When data has been scraped, run the application with ```$ python application.py```, which hosts the dashboard application locally. 

Difficulties installing GDAL. explain how to solve it tbc
