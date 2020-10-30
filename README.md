# Road-Traffic-Prediction

## Dataset Link ::  

https://data.world/cityofaustin/et93-wr2y/workspace/file?filename=traffic-studies-speed-reports-beta-1.csv 

## Dependencies 

1. Panda
2. Numpy
3. Sklearn
4. matplotlib

### Installation instructions

_Run the commands in a terminal or command-prompt.

- Install `Python 3.6 or >3.6` for your operating system, if it does not already exist.

 - For [Mac](https://www.python.org/ftp/python/3.6.8/python-3.6.8-macosx10.9.pkg)

 - For [Windows](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe)

 - For Ubuntu/Debian

 ```bash
 sudo apt-get install python3.6
 ```

 Check if the correct version of Python (3.6) is installed.

 ```bash
 python --version
 ```

**Make sure your terminal is at the root of the project i.e. where 'README.md' is located.**

* Get `virtualenv`.

 ```bash
 pip install virtualenv
 ```

* Create a virtual environment named `.env` using python `3.6` and activate the environment.

 ```bash
 # command for gnu/linux systems
 virtualenv -p $(which python3.6) .env

 source .env/bin/activate
 ```

* Install python dependencies from requirements.txt.
 ```bash
  pip install -r requirements.txt
  ```

## How To Run

After installing all the required libraries run the following commands in the terminal.

```bashres 
python3 run_1.py
 ```


# Description

* Available Features

'TRAFFIC_STUDY_SPEED_ID', 'ROW_ID', 'DATA_FILE', 'SITE_CODE', 'DATETIME', 'YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'TIME', 'SPEED_CHANNEL', 'COUNT_TOTAL', 'SPEED_0_14', 'SPEED_15_19', 'SPEED_20_24', 'SPEED_25_29', 'SPEED_30_34', 'SPEED_35_39', 'SPEED_40_44', 'SPEED_45_49', 'SPEED_50_54', 'SPEED_55_59', 'SPEED_60_64', 'SPEED_65_69', 'SPEED_70_200'


* Cluster Parameter

   Vehicle Speed  (Speed range which counted for maximum number of vehicle in one hour)
   
   

* Features Used For Training 

     * DAY_OF_MONTH
     * DAY_OF_WEEK
     * SITE_CODE
     * TIME
