# openai-pacman

Deep Reinforcement Learning with Open AI Gym – Q learning for playing Pac-Man.

## Setup

The first thing to do is to clone this repository on our computer:

```
$ git clone https://github.com/rocammo/openai-pacman.git
```

> Note: It is necessary to have `git` installed.

After cloning, a folder will be created. Inside it, you need to install virtualenv using the Python (pip) package installer:

```
# macOS
$ pip3 install virtualenv

# Debian, Ubuntu, Fedora
$ sudo pip install virtualenv
```

> Note: It is necessary to have `python3` installed.

To create the virtual environment, simply run the `virtualenv` command as follows:

```
$ virtualenv env --python=python3
```

To activate the virtual environment, run the virtualenv `activate` script installed in the `bin/` directory:

```
$ cd env
$ source bin/activate
(env)$
```

After activating it, all that is missing is to install the necessary packages (requirements.txt) using the pip packages installer:

```
(env)$ pip install -r requirements.txt
```

## Usage

To execute the project:

```
$ python main.py
```
