import sys
import os
from os.path import dirname
cwd = os.path.dirname(os.path.realpath(__file__))
rootdir = dirname(dirname(dirname(dirname(cwd))))
sys.path.append( rootdir )

import ftplib


def upload(filehandler, xpsubdir, filename):
    session = ftplib.FTP('23.233.203.221','bombardier','blackneuron')
    session.cwd( xpsubdir )
    session.storbinary('STOR '+filename, filehandler)     
    session.quit()

def fileexists(pathname, filename):
    session = ftplib.FTP('23.233.203.221','bombardier','blackneuron')
    session.cwd( pathname ) 
    L = session.nlst()
    session.quit()
    return filename in L
