# Road-Traffic-Prediction

## Dataset Link ::  https://data.world/cityofaustin/et93-wr2y/workspace/file?filename=traffic-studies-speed-reports-beta-1.csv 

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

## How To Run

After installing all the required libraries run the following commands in the terminal.

```bash
python3 run_1.py
 ```
