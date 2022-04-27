import sys
import os

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
