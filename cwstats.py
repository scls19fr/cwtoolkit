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
import random
from optparse import OptionParser
import numpy as np
import scipy.signal as signal
from scikits.audiolab import wavread, wavwrite
from hashlib import sha1

version = "0.2b, August 2014"
default_tolerance = 0.30
default_filter_multiple = 0.05 # dominant frequency +/- (this * dom. freq) defines filter width
numtaps = 10 # for the filters.  just a guess.

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
	parser = OptionParser()
	parser.add_option("-i", "--input", dest="wavfile", help="input wav file")
	parser.add_option("-t", "--tolerance", dest="tolerance", help="signal level tolerance (adjust until the number of output codes matches the number of codes in the audio recording)")
	parser.add_option("-f", "--frequency", dest="frequency", help="dominant frequency (Hz)")
	parser.add_option("-m", "--morse", action="store_true", dest="show_morse", help="show morse output")
	parser.add_option("-o", "--outstats", dest="statfile", help="write statistics to STATFILE, for use in cwtx.py")
	parser.add_option("-s", "--stats", action="store_true", dest="stats", help="show statistics")
	parser.add_option("-0", "--zero", dest="minzero", help="Use if you have a problem decoding.  Postprocessing min. allowed void size (ms).  An example might be 3 ms.")
	parser.add_option("-a", "--align", dest="realign", help="vertically realign signal by argument (keep it small)")
	parser.add_option("-A", "--amplify", dest="amplify", help="amplify (multiply) the signal by this number before filtering")
	parser.add_option("-n", "--nofilter", action="store_true", dest="nofilter", help="do not filter the signal before processing")
	parser.add_option("-w", "--width_mul", dest="width_mul", help="filter width multiplier: width = dom. freq +/- (dom. freq * width multiplier)")
	parser.add_option("-p", "--parseout", dest="parsed_file", help="save the filtered and amplified input file")
	parser.add_option("-S", "--stdin", action="store_true", dest="stdin", help="get input from stdin instead of a file")
	parser.add_option("-c", "--covert_stats", dest="cstats", help="file with statistics, needed to recover covert message")
	parser.add_option("-k", "--key", dest="code_key", help="key for use in recovering a covert message")
	parser.add_option("-v", "--version", action="store_true", dest="showversion", help="show version information and exit")
	(options, args) = parser.parse_args()

	if options.showversion:
		print "cwstats version", version
		print "Joshua Davis (cwstats@covert.codes)"
		exit()
	if not options.wavfile and not options.stdin:
		print "** Must specify an input file with -o or use the -z option"
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
	if options.width_mul:
		filter_multiple = float(options.width_mul)
	else:
		filter_multiple = float(default_filter_multiple)
	if options.statfile:
		statfile = options.statfile
	if options.code_key:
		if not options.cstats:
			print "** Must use the -c option when decoding a coded message"
			exit()

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

	if options.cstats:
		if not options.code_key:
			print "** Must use the -c option with the -k option"
			exit()

		try:
			f = open(options.cstats, "r")
		except:
			print "** Could not open", options.cstats, "for reading"
			exit()

		statstr = f.read()
		f.close()
		print "** Read statistics for covert demodulation from", options.cstats

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

	if options.amplify:
		print "** Amplifying signal by", options.amplify
		i = 0
		for d in data:
			data[i] = d * float(options.amplify)
			i = i + 1

	if not options.frequency:
		# get frequency information
		freqs = np.fft.fft(data)
		fdomain = np.fft.fftfreq(len(data))
		index = np.argmax(np.abs(freqs)**2)
		fc = abs(fdomain[index] * fs)
	else:
		fc = float(options.frequency)
	print "** Using dominant frequency:", int( round(fc) ), "Hz"

	if fc == 0:
		print "** WARNING: Dominant frequency is zero.  Vertically realigning the signal with -a may fix this.  You can also specify a dominant frequency with -f.  Exiting."
		exit()

	if not options.nofilter:
		# filter the signal
		f_nyq = fs/2
		f_cutoff_h = int(fc + (fc*filter_multiple)) # hz
		f_cutoff_l = int(fc - (fc*filter_multiple))
		print "** Filtering to between", f_cutoff_l, "and", f_cutoff_h, "Hz"

		# band-pass
		fir_coeff = signal.firwin(numtaps, [f_cutoff_l/f_nyq, f_cutoff_h/f_nyq], pass_zero=False)
		filtered_signal = signal.lfilter(fir_coeff, 1.0, data)
	else:
		filtered_signal = data

	data = filtered_signal

	if options.parsed_file:
		wavwrite(filtered_signal, options.parsed_file, fs)

	# turn the signal into rectangles
	rects = list()
	found = 0
	winsz = int( math.ceil(fs/fc) )
	winnum = int( math.floor(len(data)/winsz) )
	remainder = len(data) % winsz
	for i in range(winnum):
		for m in range(winsz):
			index = (i * winsz) + m
			if( (abs(data[index])) > tolerance):
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

	if options.minzero:
		print "** Filling gaps smaller than", options.minzero, "ms"

		minzero = options.minzero
		samples_per_ms = fs/1000
		min_zero_samples = int( math.ceil(samples_per_ms * int(minzero)) )
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
	ref = min(voids)
	if starts_with == 0: # skip any initial void
		s = 1
	else:
		s = 0
	
	ref = min(voids)
	for i in range(s, len(voids)):
		if voids[i] > 5*ref:
			void_l_ref = voids[i]
			lvoids.append(voids[i])
		elif voids[i] > 2*ref:
			void_m_ref = voids[i]
			mvoids.append(voids[i])
		else:
			void_s_ref = voids[i]
			svoids.append(voids[i])
	
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

	if options.code_key:
		decode(symbols, options.code_key, statstr, msg)
	print ""

	print "** Finished"
	print ""

def morsestats(input):
	if not input:
		return float(0), float(0)
	avg = np.average(input)
	sd = np.std(input)

	return avg, sd

def decode(symbols, key, statstr, msg):
	floats = map(float, statstr.split(","))
	dotavg = floats[0]
	dotsd = floats[1]
	dashavg = floats[2]
	dashsd = floats[3]
	inter_void_avg = floats[4]
	inter_void_sd = floats[5]
	letter_void_avg = floats[6]
	letter_void_sd = floats[7]
	word_void_avg = floats[8]
	word_void_sd = floats[9]

	# seed the PRNG
	seed = sha1(key.strip()).hexdigest()
	s = 0
	for x in seed:
		s = s + int(ord(x))
	np.random.seed(s)

	both = 0
	randoms = list()
	for m in msg:	
		if m == '.':
			avg = dotavg
			sd = dotsd

		elif m == '-':
			avg = dashavg
			sd = dashsd

		elif m == '0':
			avg = inter_void_avg
			sd = inter_void_sd

		elif m == "00":
			avg = letter_void_avg
			sd = letter_void_sd

		elif m == "000":
			avg = word_void_avg
			sd = word_void_sd


		# caveat: in the transmitter's randoms array, every 00 is preceded by a 0,
		# and every 000 is preceded by a 00.
		if sd != 0:
			if m == "000":
				tmpavg = letter_void_avg
				tmpsd = letter_void_sd
				wait = (np.random.normal(tmpavg, tmpsd) % (2*tmpsd)) + (tmpavg-tmpsd)
				both = 1

			if m == "00" or both == 1:
				tmpavg = inter_void_avg
				tmpsd = inter_void_sd
				wait = (np.random.normal(tmpavg, tmpsd) % (2*tmpsd)) + (tmpavg-tmpsd)
				both = 0

			wait = (np.random.normal(avg, sd) % (2*sd)) + (avg-sd)
			# only need the symbol randoms and the letter space
			if m == '.' or m == '-':
				randoms.append(wait)

	outmsg = list()
	for i in range(len(symbols)):
		if abs(randoms[i] - symbols[i]) > (1.5 * sd): # word seperation
			outmsg.append("000")

		elif abs(randoms[i] - symbols[i]) < sd/3: # letter seperation
			outmsg.append("00")

		elif randoms[i] > symbols[i]:
			outmsg.append('-')

		else:
			outmsg.append('.')

	letter = list()
	decoded_message = list()
	i = 0
	while i < len(outmsg):
		if outmsg[i] == '00' or outmsg[i] == '000':
			try:
				alpha = morse[''.join(letter)]
			except:
				alpha = ' '

			decoded_message.append(alpha)

			if outmsg[i] == '000':
				decoded_message.append(' ')

			letter[:] = []
			i = i + 1
			continue

		letter.append(outmsg[i])
		i = i + 1

	decoded_message = ''.join(decoded_message)

	# remove trailing '?'s (not applicable now)
	i = len(decoded_message)-1
	while i != 0:
		if decoded_message[i] == '?':
			decoded_message = decoded_message[0:i]
			i = i - 1
		else:
			break

	print ""
	print "Decoded:", ''.join(decoded_message)

main()

