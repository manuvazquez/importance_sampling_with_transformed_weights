import os
import socket
import time


def filename_from_host_and_date():

	# the name of the machine running the program (supposedly, using the socket module gives rise to portable code)
	hostname = socket.gethostname()

	# date and time
	date = time.strftime("%a_%Y-%m-%d_%H:%M:%S")

	return hostname + '_' + date + '_' + str(os.getpid())
