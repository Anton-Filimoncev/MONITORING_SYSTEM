import streamlit as st
import pandas as pd

from git import Repo

PATH_OF_GIT_REPO = "https://github.com/Anton-Filimoncev/MONITORING_SYSTEM.git"
COMMIT_MESSAGE = 'comment from python script'

def git_push():
    # try:
    repo = Repo(search_parent_directories=True)
    repo.git.add(update=True)
    repo.index.commit(COMMIT_MESSAGE)
    origin = repo.remote(name='origin')
    origin.push()
    # except:
    #     print('Some error occured while pushing the code')

git_push()