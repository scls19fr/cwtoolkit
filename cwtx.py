#!/usr/bin/python2.7
#
# Joshua Davis (cwstats@covert.codes)
# cwtx - transmit morse code with parameters and statistical attributes
#
# Copyright (C) 2014, Joshua Davis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

from __future__ import division

import sys
from os import getpid
import time
import math
from optparse import OptionParser
import numpy as np
import scipy.signal as signal
from scikits.audiolab import wavread, wavwrite
from hashlib import sha1

version = "0.2b, August 2014"
encoding = "pcm16"
default_fs = 8000 # hz
default_fc = 900 # hz
default_amplitude = .5 # max 1
default_wpm = 20
front_buffer = 2000 # quiet to append to the beginning of the output, in ms
back_buffer = 2000 # quiet for the end, ms
randoms = list()
compensated = 0
w_compensated = 0

morse = {
	".-" : "a", "-..." : "b", "-.-." : "c", "-.." : "d",
	"." : "e", "..-." : "f", "--." : "g", "...." : "h",
	".." : "i", ".---" : "j", "-.-" : "k", ".-.." : "l",
	"--" : "m", "-." : "n", "---" : "o", ".--." : "p",
	"--.-" : "q", ".-." : "r", "..." : "s", "-" : "t",
	"..-" : "u", "...-" : "v", ".--" : "w", "-..-" : "x",
	"-.--" : "y", "--.." : "z", ".----" : "1", "..---" : "2",
	"...--" : "3", "....-" : "4", "....." : "5", "-...." : "6",
	"--..." : "7", "---.." : "8", "----." : "9", "-----" : "0",
	".--.-." : "@", "---." : "!", "--..--" : ",", ".-.-.-" : ".",
	"..--.." : "?"
}

def main():
	global statfile
	global stats
	global csyms
	global cvoids
	global code_index
	statfile = None
	stats = dict()
	csyms = None
	cvoids = None
	code_index = 0

	parser = OptionParser()
	parser.add_option("-m", "--message", dest="message", help="message to encode into Morse code")
	parser.add_option("-n", "--noise", dest="noise", help="add noise to the output.  must be formatted: -n avg,sd - no spaces.  you probably want avg to be 0.")
	parser.add_option("-o", "--output", dest="outfile", help="output wav file")
	parser.add_option("-a", "--amplitude", dest="amplitude", help="output amplitude (0 <= amplitude <= 1)")
	parser.add_option("-f", "--freq", dest="cwfreq", help="cw frequency (e.g. 900), in Hz")
	parser.add_option("-F", "--sample_freq", dest="sample_freq", help="sample frequency (e.g. 8000), in Hz")
	parser.add_option("-s", "--statfile", dest="statfile", help="read message statistics from STATFILE and characterize the output wav according to the input statistics.  See README")
	parser.add_option("-w", "--wpm", dest="wpm", help="words per minute")
	parser.add_option("-S", "--stdout", action="store_true", dest="stdout", help="send to stdout instead of a file")
	parser.add_option("-c", "--covert", dest="covert_msg", help="send a covert message over the carrier message")
	parser.add_option("-k", "--key", dest="key", help="key for use in encoding the covert message")
	parser.add_option("-v", "--version", action="store_true", dest="showversion", help="show version information and exit")
	(options, args) = parser.parse_args()

	if options.stdout:
		# don't print messages to stdout if writing file to stdout
		stdoutold = sys.stdout
		sys.stdout = open('/dev/null', 'w')
	if options.showversion:
		print "cwtx version", version
		print "Joshua Davis (cwstats@covert.codes)"
		exit()
	if options.message:
		message = options.message.lower()
		message = message.strip()
	else:
		print "** Must specify input message"
		parser.print_help()
		exit()
	if not options.outfile and not options.stdout:
		print "** Must specify output wav file with -o or use the -z switch"
		parser.print_help()
		exit()
	if options.outfile and options.stdout:
		print "** Must use only one of -o or -z"
		exit()
	if options.outfile:
		outfile = options.outfile
		print "** Saving output to", outfile
	else:
		print "** Sending output to stdout"
	if options.cwfreq:
		fc = float(options.cwfreq)
	else:
		fc = default_fc
	print "** Using coding frequency", fc, "Hz"

	if options.sample_freq:
		fs = options.sample_freq
	else:
		fs = default_fs
	print "** Using sampling frequency", fs, "Hz"

	if options.wpm:
		wpm = int(options.wpm)
	else:
		wpm = default_wpm

	if options.amplitude:
		if options.amplitude < 0 or options.amplitude > 1:
			print "** Amplitude must be greater than zero, less than one."
			exit()
		amplitude = options.amplitude
	else:
		amplitude = default_amplitude

	if options.covert_msg:
		covert_msg = options.covert_msg.lower()
		covert_msg = covert_msg.strip()
		if len(covert_msg) > len(message):
			print "** Covert message must be shorter than the carrier message"
			exit()

		if not options.key:
			print "** Must include a key with the -k option when using -c"
			exit()
		if not options.statfile:
			print "** Must use the -s option when transmitting a covert message."
			exit()
	
	if options.statfile:
		if options.wpm:
			print "** Both wpm and stat file given, using stat file"

		statfile = options.statfile
		try:
			f = open(statfile)
		except:
			print "** Could not open", statfile, "for reading"
			exit()

		print "** Getting statistics from", statfile
		s = np.genfromtxt(statfile, delimiter=',', dtype=None)
		f.close()

		stats = {"dotavg" : round(s[0],2), "dotsd" : round(s[1],2), "dashavg" : round(s[2],2),
			"dashsd" : round(s[3],2), "inter_void_avg" : round(s[4],2), "inter_void_sd" : round(s[5],2),
			"letter_void_avg" : round(s[6],2), "letter_void_sd" : round(s[7],2),
			"word_void_avg" : round(s[8],2), "word_void_sd" : round(s[9],2) }

		if stats["dotavg"] == 0 or stats["dotsd"] == 0:
			print "** Provided statistics file ust have at least a dot average and dot standard deviation"
			exit()
		if stats["dashavg"] == 0:
			print "** No dash average, deriving from dot"
			stats["dashavg"] = stats["dotavg"] * 3
		if stats["inter_void_avg"] == 0:
			print "** No inter-symbol average, deriving from dot"
			stats["inter_void_avg"] = stats["dotavg"]
		if stats["letter_void_avg"] == 0:
			print "** No inter-letter average, deriving from dot"
			stats["letter_void_avg"] = stats["dotavg"] * 3
		if stats["word_void_avg"] == 0:
			print "** No inter-word average, deriving from dot"
			stats["word_void_avg"] = stats["dotavg"] * 7

	else: # not using a stats file
		print "** Speed:", wpm, "wpm"
		# element time based on http://www.kent-engineers.com/codespeed.htm
		elements_per_minute = wpm * 50 # 50 codes in PARIS
		element_ms = float(60 / elements_per_minute) * 1000 # 60 seconds
		print "** Element unit is", element_ms, "milliseconds long."

		stats["dotavg"] = element_ms
		stats["dotsd"] = 0
		stats["dashavg"] = 3 * element_ms
		stats["dashsd"] = 0
		stats["inter_void_avg"] = element_ms
		stats["inter_void_sd"] = 0
		stats["letter_void_avg"] = 3 * element_ms
		stats["letter_void_sd"] = 0
		stats["word_void_avg"] = 7 * element_ms
		stats["word_void_sd"] = 0

	if options.covert_msg:
		csyms, cvoids = mkmorse(covert_msg, 1)

	if options.key:
		seed = sha1(options.key.strip()).hexdigest()
		s = 0
		for x in seed:
			s = s + int(ord(x))
		np.random.seed(s)

	symbols, voids = mkmorse(message, 0)

	print "** Generating audio"
	output = list()
	for x in range(int(front_buffer * (fs/1000))): # give some initial void to the output
		output.append(0)

	index = x

	for i in range(len(symbols)):
		limit = index + int(symbols[i] * (fs/1000))
		for x in range(index, limit):
			output.append(amplitude * math.sin(2.0 * np.pi * fc * (x % fc/fs)))
			index = index + limit

		if i < len(symbols)-1:
			limit = index + int(voids[i] * (fs/1000))

			for x in range(index, limit):
				output.append(0)

			index = limit

	# ending void
	for x in range(index, index + int(back_buffer * (fs/1000))):
		output.append(0)

	output = np.array(output, dtype=float)

	if options.noise:
		try:
			options.noise.index(',')
		except:
			print "** -n argument requires average and standard deviation seperated by a comma, e.g. -n 0,2 - no spaces"
			exit()

		noise_avg, noise_sd = options.noise.split(',')

		# save the prng state
		r = np.random(RandomState())
		state = r.get_state()
		noise = np.random.normal(noise_avg, noise_sd, len(output))
		r.set_state(state)

		# combine the gaussian noise with the output signal
		for i in range(len(output)):
			output[i] = output[i] + noise[i]

	if not options.outfile:
		outfile = "/tmp/cwtx"+str(getpid())

	wavwrite(output, outfile, fs=fs, enc=encoding)

	if not options.outfile:
		sys.stdout = stdoutold
		f = open(outfile, "r")
		out = f.read()
		f.close()

		sys.stdout.write(out)
		sys.stdout.flush()

		exit()

	print "** Finished."
	print ""

def mktime(avg, sd, symbol):
	global statfile
	global stats
	global csyms
	global cvoids
	global code_index
	global compensated
	global w_compensated
	global randoms

	if sd == 0:
		wait = avg
	else:
		wait = (np.random.normal(avg, sd) % (2*sd)) + (avg-sd)
		if symbol == '.' or symbol == '-':
			randoms.append(wait)

	if csyms != None and code_index < len(csyms): # if there is a covert message (still)
		if symbol != '.' and symbol != '-': # don't modulate voids in the carrier
			return wait

		if cvoids[code_index-1] > 5 * stats["dotavg"] and code_index != 0: # take care of word seperation
			if w_compensated == 0:
				w_compensated = 1
				return wait + (2 * sd)
			else:
				w_compensated = 0
		elif cvoids[code_index-1] > 2 * stats["dotavg"] and code_index != 0: # end of a letter
			if compensated == 0:
				compensated = 1
				return wait
			else:
				compensated = 0

		if csyms[code_index] > 2 * stats["dotavg"]: # dash
			wait = wait - sd
			prev = '-'
		else: # dot
			wait = wait + sd
			prev = '.'

		code_index = code_index + 1

	return wait

def mkmorse(message, is_coded):
	global statfile
	global stats

	symbols = list()
	voids = list()
	inv_morse = {v:k for k, v in morse.items()}

	for e in message:
		if e != ' ':
			try:
				m = inv_morse[e]
			except:
				print "** Unknown character", e, "in message.  Update the dictionary."
				exit()

			for c in m:
				if c == '.':
					if statfile:
						if is_coded == 1:
							sd = 0
						else:
							sd = stats["dotsd"]

						dot = mktime(stats["dotavg"], sd, ".")

						if is_coded == 1:
							sd = 0
						else:
							sd = stats["inter_void_sd"]

						inter_void = mktime(stats["inter_void_avg"], sd, "0")
					else:
						dot = stats["dotavg"]
						inter_void = stats["inter_void_avg"]

					symbols.append(dot)
					voids.append(inter_void)

				elif c == '-':
					if statfile:
						if is_coded == 1:
							sd = 0
						else:
							sd = stats["dashsd"]

						dash = mktime(stats["dashavg"], sd, "-")

						if is_coded == 1:
							sd = 0
						else:
							sd = stats["inter_void_sd"]

						inter_void = mktime(stats["inter_void_avg"], sd, "0")
					else:
						dash = stats["dashavg"]
						inter_void = stats["inter_void_avg"]

					symbols.append(dash)
					voids.append(inter_void)

				else:
					print "** Bad symbol in dictionary lookup for", m
					exit()

			voids.pop(-1) # remove last inter_void
			if statfile:
				if is_coded == 1:
					sd = 0
				else:
					sd = stats["letter_void_sd"]

				letter_void = mktime(stats["letter_void_avg"], sd, "00")
			else:
				letter_void = stats["letter_void_avg"]

			voids.append(letter_void)
		else: # space
			voids.pop(-1) # remove last letter_void
			if statfile:
				if is_coded == 1:
					sd = 0
				else:
					sd = stats["word_void_sd"]
				word_void = mktime(stats["word_void_avg"], sd, "000")
			else:
				word_void = stats["word_void_avg"]
			voids.append(word_void)

	voids.pop(-1) # take away the last void

	return (symbols, voids)

main()

