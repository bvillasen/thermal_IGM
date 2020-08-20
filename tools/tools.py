import os, sys
from os import listdir
from os.path import isfile, join



def create_directory( dir ):
  print(("Creating Directory: {0}".format(dir) ))
  indx = dir[:-1].rfind('/' )
  inDir = dir[:indx]
  dirName = dir[indx:].replace('/','')
  dir_list = next(os.walk(inDir))[1]
  if dirName in dir_list: print( " Directory exists")
  else:
    os.mkdir( dir )
    print( " Directory created")

