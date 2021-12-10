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

### False
- Hilary Clinton is the president
- Climate change is a hoax


## References

@article{shu2018fakenewsnet,
  title={FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media},
  author={Shu, Kai and  Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={arXiv preprint arXiv:1809.01286},
  year={2018}
}

@article{shu2017fake,
  title={Fake News Detection on Social Media: A Data Mining Perspective},
  author={Shu, Kai and Sliva, Amy and Wang, Suhang and Tang, Jiliang and Liu, Huan},
  journal={ACM SIGKDD Explorations Newsletter},
  volume={19},
  number={1},
  pages={22--36},
  year={2017},
  publisher={ACM}
}

@article{shu2017exploiting,
  title={Exploiting Tri-Relationship for Fake News Detection},
  author={Shu, Kai and Wang, Suhang and Liu, Huan},
  journal={arXiv preprint arXiv:1712.07709},
  year={2017}
}
