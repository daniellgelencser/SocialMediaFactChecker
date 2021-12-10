# Social Media Fact Checker

## Run application

_The examples has been run on Manjaro Linux, equivalent commands may be used with other Linux distributions._

1. Clone repository
```git clone https://github.com/daniellgelencser/SocialMediaFactChecker.git```
2. Donwload the models from the following link: https://we.tl/t-zMTHkYmytf (active until 9 March 2022)
3. Move the model in the following directory: ```<project root>/SocialMediaFactChecker/MachineLearning/```
4. Create virtual environment with Python 3.9 executable \
```virtualenv /usr/bin/python3.9 venv```
5. Activate virtual environment \
```source <project root>/venv/bin/activate```
6. Install dependencies \
```pip install -r <project root>/requirements.txt ```
7. Run web server \
```cd <project root>/SocialMediaFactChecker``` \
```python manage.py runserver```
8. Open a browser and go to http://127.0.0.1:8000/FactChecker/


## Example inputs

### True
- Americans evacuated from Wuhan
- Hurricane Laura strikes in Louisiana
- Amy Coney Barrett confirmed to the supreme court
