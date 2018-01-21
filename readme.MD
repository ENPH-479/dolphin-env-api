# **Dolphin Emulator Environment API**

<p align="center">
  <img src="./Demo.gif"/>
</p>

## What is it? 
This project aims to provide modular, reusable code for programmatically interfacing with the [Dolphin emulator](https://dolphin-emu.org/). The main motivation behind the project is to provide code that makes it easier for [artificial intelligence (AI)](https://en.wikipedia.org/wiki/Artificial_intelligence) agents to interact with emulated video games, whether the end user is an AI researcher or just someone looking for a fresh twist on "playing" video games. 

## How do I use it? 
### 1. Pipe Setup

Assuming Dolphin has already been installed on Linux or MacOS, setup automated [pipe](https://en.wikipedia.org/wiki/Pipeline_(Unix)) inputs through the provided script, or by [following the instructions here](https://wiki.dolphin-emu.org/index.php?title=Pipe_Input).

To use the provided setup script, execute:

```
./setup.sh
```
### 2. Using Pipenv

Install [Pipenv](https://docs.pipenv.org/) and initialize the environment by executing: 
```
pip install pipenv
pipenv install --three
```

Run the python [virtualenv](https://virtualenv.pypa.io/en/stable/) by executing:

```
pipenv shell
```

## Caveats
* As of December 6th/2017, the project is being developed and tested exclusively on the [Debian](https://www.debian.org/) 9 [Linux](https://en.wikipedia.org/wiki/Linux) operating system, though portions of the code have been tested on MacOS and [Ubuntu](https://www.ubuntu.com/)
* This codebase and its documentation are still a work in progress

### Team

* Ruishen Lu
* Brian Irwin
* Andy Tu

### History
* This project was initiated as part of the ENPH 479 engineering capstone project course at the University of British Columbia (UBC)
