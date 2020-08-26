import os, sys
from os import listdir
from os.path import isfile, join

def printProgress( current, total,  deltaTime, print_str='' ):
  terminalString = "\rProgress: "
  if total==0: total+=1
  percent = 100.*current/total
  nDots = int(percent/5)
  dotsString = "[" + nDots*"." + (20-nDots)*" " + "]"
  percentString = "{0:.0f}%".format(percent)
  ETR = deltaTime/(current+1)*(total - current)
  hours = int(ETR/3600)
  minutes = int(ETR - 3600*hours)//60
  seconds = int(ETR - 3600*hours - 60*minutes)
  ETRstring = "  ETR= {0}:{1:02}:{2:02}    ".format(hours, minutes, seconds)
  if deltaTime < 0.0001: ETRstring = "  ETR=    "
  terminalString  += dotsString + percentString + ETRstring
  terminalString += print_str
  sys.stdout. write(terminalString)
  sys.stdout.flush() 

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

