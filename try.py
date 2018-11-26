from pymedtermino import *
from pymedtermino import snomedct as sn

pymedtermino.DATA_DIR = "D:/Lecture Notes/Semester 2/Text Mining/CA - 1/codebase/PyMedTermino-0.3.3"
pymedtermino.LANGUAGE = 'en'

words = sn.SNOMEDCT.search("Knee")

print(words)
