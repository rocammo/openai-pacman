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

### Virtual display for Ubuntu subsystem for Windows

If you are using the Ubuntu subsystem for Windows you will encounter this error if you try to render the simulation.

```
pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
```

This is caused by not having a display attached to your terminal, it can be fixed by installing a X server on Windows, we recommend
[Xming X Server for Windows](https://sourceforge.net/projects/xming/) as is easy to install and use.

Then just run the following command on the linux terminal to attach the virtual display:

```
export DISPLAY=:0
```

This setting will only persist on the current terminal session so you can place it on .bashrc to make it persisten.
Remember that Xming server should be running on Windows, as is needed to handle the output of your Ubuntu subsystem.

## Usage

To execute the project in training mode:
The training mode starts from scratch (epsilon = 1)
```
$ python -m pacman -n DDQN -m TRAIN
```

To execute the project in testing mode:
```
$ python -m pacman -n DDQN -m TEST -p ./openai-pacman/results/cpu-1200-episodes/
```
The testing mode loads a file and starts from that knowledge represented in weights (epsilon near 0).

## Algorithms

Our project implements three different reinforcement learning algorithms in the same framework:

 * [x] Deep Q Reinforcement Learning
 * [x] Duel Q Reinforcement Learning
 * [x] Cross Entropy
 
You can choose the algorithm to run by using the `--network` parameter.
 
## Screenshots

<img width="272" alt="screenshot" src="https://user-images.githubusercontent.com/9489977/67415907-26839c80-f5c6-11e9-830d-a39ad2d13dd2.png">
