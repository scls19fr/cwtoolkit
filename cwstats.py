#!/usr/bin/python2.7
#
# Joshua Davis (cwstats@covert.codes)
# cwstats - decode morse code and optionally generate statistics
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
import matplotlib.pyplot as plt

version = "0.1b, July 2014"
default_tolerance = 0.30
default_minzero = 40 # ms.  sequences of zeros less than this are filled in before decoding
default_filter_multiple = 0.3 # dominant frequency +/- (this * dom. freq) defines filter width
numtaps = 100 # for the filters.  just a guess.

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
	parser.add_option("-i", "--input", dest="wavfile", help="input wav file")
	parser.add_option("-t", "--tolerance", dest="tolerance", help="signal level tolerance (adjust until the number of output codes matches the number of codes in the audio recording)")
	parser.add_option("-f", "--frequency", dest="frequency", help="dominant frequency (Hz)")
	parser.add_option("-m", "--morse", action="store_true", dest="show_morse", help="show morse output")
	parser.add_option("-o", "--outfile", dest="statfile", help="write statistics to STATFILE, for use in cwtx.py")
	parser.add_option("-s", "--stats", action="store_true", dest="stats", help="show statistics")
	parser.add_option("-0", "--zero", dest="minzero", help="min. allowed void size (ms)")
	parser.add_option("-a", "--align", dest="realign", help="vertically realign signal by argument (keep it small)")
	parser.add_option("-n", "--nofilter", action="store_true", dest="nofilter", help="do not filter the signal before processing")
	parser.add_option("-w", "--width_mul", dest="width_mul", help="filter width multiplier: width = dom. freq +/- (dom. freq * width multiplier)")
	parser.add_option("-z", "--stdin", action="store_true", dest="stdin", help="get input from stdin instead of a file")
	parser.add_option("-v", "--version", action="store_true", dest="showversion", help="show version information and exit")
	(options, args) = parser.parse_args()

	if options.showversion:
		print "cwstats version", version
		print "Joshua Davis (cwstats@covert.codes)"
		exit()
	if not options.wavfile and not options.stdin:
		print "** Must specify input file with -o or use the -z option"
		print ""
		parser.print_help()
		exit()
	if options.wavfile and options.stdin:
		print "** Must us only one of -o or -z"
		exit()
	if not options.tolerance:
		tolerance = float(default_tolerance)
	else:
		tolerance = float(options.tolerance)
	if options.minzero:
		minzero = float(options.minzero)
	else:
		minzero = float(default_minzero)
	if options.width_mul:
		filter_multiple = float(options.width_mul)
	else:
		filter_multiple = float(default_filter_multiple)
	if options.statfile:
		statfile = options.statfile

	if options.stdin:
		print "** Reading from stdin"
		data = sys.stdin.read()

		outfile = "/tmp/cwstats"+str(getpid())
		f = open(outfile, 'w')
		f.write(data)
		f.close()

		wavfile = outfile
	else:
		wavfile = options.wavfile

	data, fs, encoding = wavread(wavfile)
	print ""
	print "** Read", len(data), "samples from", wavfile, "with sample frequency", fs, "Hz and encoding", encoding

	print "** Signal average:", np.average(data)

	if options.realign:
		print "** Vertically realigning signal by", options.realign
		for i in range(len(data)):
			data[i] = data[i] + float(options.realign)
		print "** Signal average:", np.average(data)

	# want mono
	mono = 0
	try:
		x = data[0][1]
	except:
		mono = 1

	if mono == 0:
		print "** Converting to mono"
		tmp = list()
		for d in data:
			tmp.append(d[0])

		data = np.array(tmp)

	# get frequency information
	freqs = np.fft.fft(data)
	fdomain = np.fft.fftfreq(len(data))
	index = np.argmax(np.abs(freqs)**2)

	if not options.frequency:
		dfreq = abs(fdomain[index] * fs)
	else:
		dfreq = float(options.frequency)
	print "** Using dominant frequency:", int( round(dfreq) ), "Hz"

	if dfreq == 0:
		print "** WARNING: Dominant frequency is zero.  Vertically realigning the signal with -a may fix this.  You can also specify a dominant frequency with -f.  Exiting."
		exit()

	print "** Using min. zero (gap) time:", minzero

	if not options.nofilter:
		# filter the signal
		f_nyq = fs/2
		f_cutoff_h = dfreq + (dfreq*filter_multiple) # hz
		f_cutoff_l = dfreq - (dfreq*filter_multiple)
		print "** Filtering to between", f_cutoff_l, "and", f_cutoff_h

		# band-pass
		fir_coeff = signal.firwin(numtaps, [f_cutoff_l/f_nyq, f_cutoff_h/f_nyq], pass_zero=False)
		filtered_signal = signal.lfilter(fir_coeff, 1.0, data)
	else:
		filtered_signal = data

	data = filtered_signal

	# turn the signal into rectangles
	rects = list()
	found = 0
	winsz = int( math.ceil(fs/dfreq) )
	winnum = int( math.floor(len(data)/winsz) )
	remainder = len(data) % winsz
	for i in range(winnum):
		for m in range(winsz):
			key = (i * winsz) + m
			if( (abs(data[key])) > tolerance):
				found = 1
				break

		if found == 1:
			num = 1
			found = 0
		else:
			num = 0

		rects.extend(num for o in range(winsz))
	
	if remainder > 0:
		for m in range(winsz * winnum, (winsz * winnum) + remainder):
			if(data[m] > tolerance):
				found = 1
				break

		if found == 1:
			num = 1
		else:
			num = 0

		rects.extend(num for o in range(winsz))

	# hack to fill the holes
	# this and the above loops might be integrated
	# the fact that gaps exists suggests that my approach above needs fixed
	samples_per_ms = int( math.floor(fs/1000) )
	min_zero_samples = int( math.ceil(samples_per_ms * minzero) )
	counter = 0
	exceeded = 0
	zeros = 0
	last_number = rects[0]
	for i in range(1, len(rects)):
		if rects[i] == 0:
			if last_number == 0:
				zeros = zeros + 1

				if zeros == min_zero_samples:
					exceeded = 1
			else:
				last_number = 0
				zeros = 1
				counter = i
				exceeded = 0
		else: # rects[i] == 1
			if last_number == 0 and exceeded == 0:
				for w in range(counter, i):
					rects[w] = 1
			last_number = 1

	starts_with = rects[0]

	# decode
	print "** Decoding with tolerance: %.3f" % tolerance

	symbols = list()
	voids = list()
	base = 0
	i = 0
	index = 0
	previous = 0
	samples = 0
	for r in rects:
		i = i + 1

		if r == 1:
			if previous == 1:
				index = index + 1
				samples = index - base # in case this is the last symbol
				continue # continuing a symbol

			else: # started a symbol, ended a void
				samples = index - base
				t = samples / (fs/1000)
				if t > 0:
					voids.append(t)

				index = base = i
				previous = 1

		else: # r == 0
			if previous == 1: # started a void, finished a symbol
				samples = index - base
				t = samples / (fs/1000)
				symbols.append(t)

				index = base = i
				previous = 0
			else: # continuing a void
				index = index + 1
				samples = index - base
				continue

	# Handle the case of a final symbol (symbol ends the wav)
	if r == 1:
		t = samples / (fs/1000)
		symbols.append(t)

	total = 0
	for v in voids:
		total = total + v
	for s in symbols:
		total = total + s

	if not symbols:
		print "** No symbols found in message."
		exit()
	if not voids:
		print "** No gaps found in message."

	# figure out what's a dot, what's a dash, what's a short void, and what's a long void
	# dashes are supposed to be 3x dot length
	# inter-element voids are 1x dot length, inter-character 3x, inter-word 7x
	dash_ref = 0
	dot_ref = 0
	ref = symbols[0]
	for i in range(1, len(symbols)):
		if symbols[i] > 2*ref:
			dash_ref = symbols[i]
		elif ref > 2*symbols[i]:
			dot_ref = symbols[i]

		ref = symbols[i]

		if dot_ref and dash_ref:
			break

	if dash_ref == 0 or dot_ref == 0:
		print "** All symbols are the same.  Assuming dots."
		if not dot_ref:
			dot_ref = symbols[0]

	void_s_ref = 0
	void_m_ref = 0
	void_l_ref = 0
	lvoids = list()
	mvoids = list()
	svoids = list()
	ref = voids[0]
	for i in range(1, len(voids)):
		if voids[i] > 5*ref:
			void_l_ref = voids[i]
			lvoids.append(voids[i])
		elif voids[i] > 2*ref:
			void_m_ref = voids[i]
			mvoids.append(voids[i])
		elif ref > 2*voids[i]:
			void_s_ref = voids[i]
			svoids.append(voids[i])
		elif ref > 5*voids[i]:
			void_l_ref = ref

		ref = voids[i]

		if void_s_ref and void_m_ref and void_l_ref:
			break
	
	if void_l_ref == 0:
		print "** No word delineation detected."

	if starts_with == 0: # remove the first void
		voids.pop(0)

	dots = list()
	dashes = list()
	msg = list()
	for i in range(0, len(symbols)):
		if symbols[i] < 2*dot_ref:
			msg.append('.')
			dots.append(symbols[i])
		else:
			msg.append('-')
			dashes.append(symbols[i])

		try:
			if voids[i] < 2 * void_s_ref:
				msg.append('0')
			elif voids[i] < 5 * void_s_ref:
				msg.append('00')
			else:
				msg.append('000')
		except: # fails at the end b/c we stripped away beginning and trailing voids
			pass

	dotavg, dotsd = morsestats(dots)
	dashavg, dashsd = morsestats(dashes)
	lvoidavg, lvoidsd = morsestats(lvoids)
	mvoidavg, mvoidsd = morsestats(mvoids)
	svoidavg, svoidsd = morsestats(svoids)

	duration = len(data)/fs
	epm = (len(symbols)*60)/duration

	if options.stats:
		print "** Message has", len(symbols), "symbols and", len(voids), "voids."
		print ""
		print "----------"
		print "| Number of dots:", len(dots)
		print "| Avg. dot length:", dotavg, "ms"
		print "| Dot stdev:", dotsd, "ms"
		print "|"	
		print "| Number of dashes:", len(dashes)
		print "| Avg. dash length:", dashavg, "ms"
		print "| Dash stdev:", dashsd, "ms"
		print "|"	
		print "| Number of intra-letter spaces", len(svoids)
		print "| Avg. intra-letter spacing length:", svoidavg, "ms"
		print "| Intra-letter spacing stdev:", svoidsd, "ms"
		print"|"
		print "| Number of inter-letter spaces:", len(mvoids)
		print "| Avg. inter-letter spacing length:", mvoidavg, "ms"
		print "| Inter-letter spacing stdev:", mvoidsd, "ms"
		print "|"	
		print "| Number of inter-word spaces:", len(lvoids)
		print "| Avg. Inter-word spacing length:", lvoidavg, "ms"
		print "| Inter-word spacing stdev:", lvoidsd, "ms"
		print "|"
		print "| Signal duration:", duration, "sec."
		print "| Approx. elements (dots, dashes) per minute:", epm
		print "----------"

	if options.statfile:
		try:
			f = open(statfile, "w")
		except:
			print "** Could not open", statfile, "for writing"

		statstr = "%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f"\
			% (dotavg, dotsd, dashavg, dashsd, svoidavg, svoidsd, mvoidavg, mvoidsd, lvoidavg, lvoidsd)
		f.write(statstr)
		f.close()
		print "** Wrote statistics to", statfile

	letter = list()
	message = list()
	i = 0
	while i < len(msg):
		if msg[i] == '0':
			i = i + 1
			continue

		if msg[i] == '00':
			# letter seperation
			try:
				alpha = morse[''.join(letter)]
			except:
				alpha = '?'

			message.append(alpha)
			letter[:] = []
			i = i + 1
			continue

		if msg[i] == '000':
			# word seperation
			try:
				alpha = morse[''.join(letter)]
			except:
				alpha = '?'

			message.append(alpha)
			letter[:] = []
			message.append(' ')
			i = i + 1
			continue

		letter.append(msg[i])
		i = i + 1

	# print out the final letter
	try:
		alpha = morse[''.join(letter)]
	except:
		alpha = '?'
	message.append(alpha)

	long_voids = list()
	medium_voids = list()
	small_voids = list()
	for v in voids:
		if v > 5 * void_s_ref:
			long_voids.append(v)
		elif v > 2 * void_s_ref:
			medium_voids.append(v)
		else:
			small_voids.append(v)

	# show morse if they want it
	if options.show_morse == 1:
		print ""
		for m in msg:
			if m == '0':
				pass
			elif m == '00':
				sys.stdout.write(' ')
			elif m == '000':
				sys.stdout.write(' [word gap] ')
			else:
				sys.stdout.write(m)

			sys.stdout.flush()
	
	print ""
	print "Message:", ''.join(message)
	print ""

def morsestats(input):
	if not input:
		return float(0), float(0)
	avg = np.average(input)
	sd = np.std(input)

	return avg, sd


main()

