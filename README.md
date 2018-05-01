# ml4aad
Machine Learning for Automated Algorithm Design

## Installation
First of all, install Python3.6 and its dev files:
```bash
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.6
$ sudo apt-get install python3.6-dev
$ sudo apt-get install python3.6-tk
```

Install `swig3.0`:
```bash
$ sudo apt-get remove swig
$ sudo apt-get install swig3.0
$ sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
```

Create a `virtualenv` with Python3.6 in the project root:
```bash
$ virtualenv -v python3.6 venv
$ source venv/bin/activate
```

Install required packages:
```bash
$ pip install Cython
$ pip install pyrfr==0.8.0 --no-cache
$ pip install numpy scipy sklearn
```

Install SMAC3 from my own repository to avoid the bug with `pyrfr`:
```bash
$ pip install git+https://github.com/angellandros/SMAC3.git@development
```
