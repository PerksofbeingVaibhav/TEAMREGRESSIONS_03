# TEAMREGRESSIONS_03

This is the AI powered- Assiant Bot -"SOPHIA", 
WHich helps in the covid-19 times and provide the people with 
the access of help of the Doctor, without meeting the Doctor in person.
It is the beginning model of the representation on, how with the help of the AI
people can ge their treatment from the home and for free, and under sever conditions only
be asked for to meet the Doctor.


It benefits the user/people to stay safe at the home and have no contact with people outside 
his home and keeps him safe from the covid-19.

It is just in it's initial phase of development, but with furthur working, 
and improving the quality of the AI-bot Sophia, we can make things easier for the Doctors 
and also the people.

Hope, our idea brings a new sense of hope, that in future, with the help of the AI-bot 
just like Sophia, we can make the life of the people easier in every aspect of life.



-----------------------------------------------------------------------

This Project is Open-Source,
So other developers can contribute in it and make it work more seemlessly.


## Installation

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```


### Activate it
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```
### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run

You also need 'FLASK'
```console
flask run
```

To run Flask in development enviornment and how Flask work, try to refer to the FLASK Documentation.(https://flask.palletsprojects.com/en/2.0.x/)

## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```
