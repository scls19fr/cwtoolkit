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
import time
import math
from optparse import OptionParser
import numpy as np
import scipy.signal as signal
from scikits.audiolab import wavread, wavwrite
import matplotlib.pyplot as plt

version = "0.1b, July 2014"
fs = 8000 # Sample rate, Hz
encoding = "pcm16"
default_fc = 900 # hz
default_amplitude = .5 # max 1
default_wpm = 20
front_buffer = 2000 # quiet to append to the beginning of the output, in ms
back_buffer = 2000 # quiet for the end, ms

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
	".--.-." : "@", "---." : "!", "--..--" : ",", ".-.-.-" : "."
}

def main():
	parser = OptionParser()
	parser.add_option("-m", "--message", dest="message", help="message to encode into Morse code")
	parser.add_option("-n", "--noise", dest="noise", help="add noise to the output.  must be formatted: -n avg,sd - no spaces.  you probably want avg to be 0.")
	parser.add_option("-o", "--output", dest="outfile", help="output wav file")
	parser.add_option("-a", "--amplitude", dest="amplitude", help="output amplitude (0 <= amplitude <= 1)")
	parser.add_option("-f", "--freq", dest="cwfreq", help="cw frequency (e.g. 900), in Hz")
	parser.add_option("-s", "--statfile", dest="statfile", help="read message statistics from STATFILE and characterize the output wav according to the input statistics.  See README.md")
	parser.add_option("-w", "--wpm", dest="wpm", help="words per minute")
	parser.add_option("-v", "--version", action="store_true", dest="showversion", help="show version information and exit")
	(options, args) = parser.parse_args()

	if options.showversion:
		print "cwtx version", version
		print "Joshua Davis (cwstats@covert.codes)"
		exit()
	if not options.message:
		print "** Must specify input message"
		parser.print_help()
		exit()
	if not options.outfile:
		print "** Must specify output wav file"
		parser.print_help()
		exit()
	if options.cwfreq:
		fc = float(options.cwfreq)
	else:
		fc = default_fc
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

	if options.statfile:
		if options.wpm:
			print "** Both wpm and stat file specifiled, using stat file"
		statfile = options.statfile

		try:
			f = open(statfile)
		except:
			print "** Could not open", statfile, "for reading"
			exit()

		input = np.genfromtxt(statfile, delimiter=',', dtype=None)
		dotavg = input[0]; dotsd = input[1]
		if dotavg == 0 or dotsd == 0:
			print "** Must have at least a dot average and standard deviation"
			exit()

		dashavg = input[2]; dashsd = input[3]
		inter_void_avg = input[4]; inter_void_sd = input[5]
		letter_void_avg = input[6]; letter_void_sd = input[7]
		word_void_avg = input[8]; word_void_sd = input[9]

		f.close()

	message = options.message.lower()
	outfile = options.outfile

	print ""
	print "** Encoding Morse code to:", outfile
	print "** Using cw frequency:", fc, "Hz"
	if options.statfile:
		print "** Using statistics from", options.statfile
	else:
		print "** Speed:", wpm, "wpm"

		# element time based on http://www.kent-engineers.com/codespeed.htm
		elements_per_minute = wpm * 50 # 50 codes in PARIS
		element_ms = float(60 / elements_per_minute) * 1000 # 60 seconds
		print "** Element unit is", element_ms, "milliseconds long."

		dot = element_ms
		dash = 3 * element_ms
		inter_void = element_ms
		letter_void = 3 * element_ms
		word_void = 7 * element_ms

	inv_morse = {v:k for k, v in morse.items()}

	message = message.strip()

	symbols = list()
	voids = list()
	for e in message:
		if options.statfile:
			dot = mktime(dotavg, dotsd)

			# the measurements wrt dot length are taken from the web, e.g. wikipedia's entry on morse code
			if dashavg == 0:
				dashavg = 3 * dotavg
			if dashsd == 0:
				dashsd = dotsd
			dash = mktime(dashavg, dashsd)

			if inter_void_avg == 0:
				inter_void_avg = dotavg
			if inter_void_sd == 0:
				inter_void_sd = dotsd
			inter_void = mktime(inter_void_avg, inter_void_sd)

			if letter_void_avg == 0:
				letter_void_avg = 3 * dotavg
			if letter_void_sd == 0:
				letter_void_sd = dotsd
			letter_void = mktime(letter_void_avg, letter_void_sd)

			if word_void_avg == 0:
				word_void_avg = 7 * dotavg
			if word_void_sd == 0:
				word_void_sd = dotsd
			word_void = mktime(word_void_avg, word_void_sd)

		if e != ' ':
			try:
				m = inv_morse[e]
			except:
				print "** Unknown character", m, "in message.  Update the dictionary."
				exit()

			for c in m:
				if c == '.':
					symbols.append(dot)
					voids.append(inter_void)
				elif c == '-':
					symbols.append(dash)
					voids.append(inter_void)
				else:
					print "** Bad character in dictionary lookup for", m
					exit()

			voids.pop(-1) # remove last inter_void
			voids.append(letter_void)
		else: # space
			voids.pop(-1) # remove last letter_void
			voids.append(word_void)

	voids.pop(-1) # take away the last void

	# make the audio file
	print "** Making output wav file"

	output = list()
	for x in range(int(front_buffer * (fs/1000))): # give some initial void to the output
		output.append(0)

	index = int(x)

	for i in range(len(symbols)):
		limit = index + int(symbols[i] * (fs/1000))
		for x in range(index, limit):
			output.append(amplitude * math.sin(2 * np.pi * fc * (x % fc/fs)))
			index = index + limit

		if i < len(symbols)-1:
			limit = index + int(voids[i]  * (fs/1000))

			for x in range(index, limit):
				output.append(0)

			index = limit

	for x in range(index, index + int(back_buffer * (fs/1000))):
		output.append(0)

	output = np.array(output, dtype=float)

	if options.noise:
		try:
			options.noise.index(',')
		except:
			print "** -n argument requires average and standard deviation seperated by a comma, e.g. -n 5,2 - no spaces"
			exit()

		noise_avg, noise_sd = options.noise.split(',')
		noise = np.random.normal(noise_avg, noise_sd, len(output))

		# combine the gaussian noise with the output signal
		for i in range(len(output)):
			output[i] = output[i] + noise[i]

	wavwrite(output, outfile, fs=fs, enc=encoding)

	print "** Finished."
	print ""

def mktime(avg, sd):
	wait = np.random.normal(avg, sd)
	return wait

main()

