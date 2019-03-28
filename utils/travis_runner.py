#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import subprocess as sp

if __name__ == '__main__':

    notebook = 'student_project.ipynb'
    cmd = ' jupyter nbconvert --execute {}  --ExecutePreprocessor.timeout=-1'.format(notebook)
    sp.check_call(cmd, shell=True)
