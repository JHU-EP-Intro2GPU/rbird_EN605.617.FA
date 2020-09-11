#!/usr/bin/python3

import _thread
import random
import sys
import time

# Define a function for the thread
def print_time( threadName, delay):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print ("%s: %s" % ( threadName, time.ctime(time.time()) ))

# Create two threads as follows
num_threads = 2

if len(sys.argv):
    num_threads = int(sys.argv[0])

try:
    for i in Range(num_threads):
       delay = random.randint(1, 4)
       _thread.start_new_thread( print_time, (f"Thread-{num_threads + 1}", delay, ) )

#       _thread.start_new_thread( print_time, ("Thread-1", 2, ) )
#       _thread.start_new_thread( print_time, ("Thread-2", 4, ) )
except:
   print ("Error: unable to start thread")

while 1:
   pass
