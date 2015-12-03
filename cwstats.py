#!/usr/bin/python2.7
#
# Joshua Davis (cwstats - covert.codes)
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
import scipy
import scipy.io.wavfile
from hashlib import sha1

version = "0.3b, August 2014"
default_tolerance = 20
maxwpm = 50 # safe to leave it at 50
filter_bw = maxwpm * 4
default_window_alts = 2 # number of fc to include in window when transforming signal->rectangles

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
    parser.add_option("-t", "--tolerance", dest="tolerance", help="signal level tolerance (usually <= 30)")
    parser.add_option("-W", "--window-alts", dest="window_alts", help="window size in alternations of carrier (increase if you have trouble detecting, e.g. to 3 or 4)")
    parser.add_option("-f", "--frequency", dest="frequency", help="dominant frequency override (Hz)")
    parser.add_option("-m", "--morse", action="store_true", dest="show_morse", help="show Morse output")
    parser.add_option("-o", "--outstats", dest="statfile", help="write statistics to STATFILE, for use in cwtx.py")
    parser.add_option("-s", "--stats", action="store_true", dest="stats", help="show statistics")
    parser.add_option("-a", "--align", dest="realign", help="vertically realign signal by argument")
    parser.add_option("-n", "--nofilter", action="store_true", dest="nofilter", help="do not filter the signal before processing.  Try this if you have problems decoding a weak signal.")
    parser.add_option("-w", "--width_mul", dest="width_mul", help="filter width multiplier: width = dom. freq +/- (dom. freq * width multiplier)")
    parser.add_option("-p", "--parseout", dest="parsed_file", help="save the filtered input file")
    parser.add_option("-S", "--stdin", action="store_true", dest="stdin", help="get input from stdin instead of a file")
    parser.add_option("-c", "--covert_stats", dest="cstats", help="file with statistics, needed to recover covert message")
    parser.add_option("-k", "--key", dest="code_key", help="key for use in recovering a covert message")
    parser.add_option("-v", "--version", action="store_true", dest="showversion", help="show version information and exit")
    (options, args) = parser.parse_args()

    if options.showversion:
        print("cwstats version %s" % version)
        print("Joshua Davis (cwstats - covert.codes)")
        exit()

    if not options.wavfile and not options.stdin:
        print("** Must specify an input file with -o or use the -z option")
        print("")
        parser.print_help()
        exit()

    if options.wavfile and options.stdin:
        print("** Must us only one of -o or -z")
        exit()

    if options.tolerance:
        tolerance = float(options.tolerance)/100
        if tolerance > 1:
            print("** Tolerance must be between 0 and 100")
            exit()
    else:
        tolerance = float(default_tolerance)/100

    if options.statfile:
        statfile = options.statfile

    if options.window_alts:
        window_alts = int(options.window_alts)
    else:
        window_alts = int(default_window_alts)

    if options.code_key or options.cstats:
        if not options.code_key or not options.cstats:
            print("** The -c and -k options must be used together")
            exit()

        try:
            f = open(options.cstats, "r")
        except:
            print("** Could not open %s for reading" % options.cstats)
            exit()

        statstr = f.read()
        f.close()
        print("** Read statistics for covert demodulation from %s" % options.cstats)

    if options.stdin:
        print("** Reading from stdin")
        data = sys.stdin.read()

        outfile = "/tmp/cwstats"+str(getpid())
        f = open(outfile, 'w')
        f.write(data)
        f.close()

        wavfile = outfile
    else:
        wavfile = options.wavfile

    fs, data = scipy.io.wavfile.read(wavfile)
    print("")

    print("** Read %d samples from %s with sample frequency %s Hz" % (len(data), wavfile, fs))

    print("** Signal average: %s" % np.average(data))

    if options.realign:
        print("** Vertically realigning signal by %s" % options.realign)
        for i in range(len(data)):
            data[i] = data[i] + float(options.realign)
        print("** Signal average: %s" % np.average(data))

    # want mono
    try:
        tmp = list()
        for d in data:
            tmp.append(d[0])
        data = np.array(tmp)
    except:
        # already mono
        pass

    if options.frequency:
        fc = float(options.frequency)
    else:
        freqs = np.fft.fft(data)
        fdomain = np.fft.fftfreq(len(data))
        index = np.argmax(np.abs(freqs)**2)
        fc = int(round(abs(fdomain[index] * fs)))

    if fc == 0:
        print("** Dominant frequency is zero.  Vertically realigning the signal with -a may fix this.  You can also specify a dominant frequency with -f.")
        exit()
    else:
        print("** Using dominant frequency: %s Hz" % fc)

    if options.nofilter:
        filtered_signal = data
    else:
        f_cutoff_h = int(fc + filter_bw/2)
        f_cutoff_l = int(fc - filter_bw/2)
        print("** Filtering to between %s and %s Hz" % (f_cutoff_l, f_cutoff_h))

        f_nyq = fs/2
        numtaps = fs/f_cutoff_l

        fir_coeff = signal.firwin(numtaps, [f_cutoff_l/f_nyq, f_cutoff_h/f_nyq], window='blackmanharris', pass_zero=False, nyq=f_nyq)
        filtered_signal = signal.lfilter(fir_coeff, 1.0, data)
        data = filtered_signal

    if options.parsed_file:
        wavwrite(filtered_signal, options.parsed_file, fs)

    # Turn the filtered signal into symbols
    rects = signal_to_rects(data, tolerance, fs, fc, window_alts)
    symbols,voids = rects_to_symbols(rects, fs)
    dots,dashes = symbols_to_beeps(symbols)

    if not dots:
        print("** No symbols found in input")
        exit()

    dotavg = np.average(dots)
    print("** Avg. dot length: %s ms" % dotavg)

    # Remove any leading or trailing voids
    if rects[0] == 0:
        voids.pop(0)
    
    print("** Decoding with tolerance: %s %%" % tolerance)

    # categorize voids, for statistics
    lvoids = list()
    mvoids = list()
    svoids = list()
    for i in range(len(voids)):
        if voids[i] > 5*dotavg:
            lvoids.append(voids[i])
        elif voids[i] > 2*dotavg:
            mvoids.append(voids[i])
        else:
            svoids.append(voids[i])
    
    if not lvoids:
        print("** No word delineation detected in message.")
    if not mvoids:
        print("** No letter delineation detected in message.")
    if not svoids:
        print("** No symbol delineation detected in message.")

    code = symbols_to_morse(symbols, voids, dotavg)

    # make the statistics
    dotavg, dotsd = morsestats(dots)
    dashavg, dashsd = morsestats(dashes)
    lvoidavg, lvoidsd = morsestats(lvoids)
    mvoidavg, mvoidsd = morsestats(mvoids)
    svoidavg, svoidsd = morsestats(svoids)
    duration = len(data)/fs
    epm = (len(symbols)*60)/duration

    if options.stats:
        print("** Message has %d symbols and %d voids." % (len(symbols), len(voids)))
        print("")
        print("----------")
        print("| Number of dots: %d" % len(dots))
        print("| Avg. dot length: %f" % dotavg)
        print("| Dot stdev: %s ms" % dotsd)
        print("|")
        print("| Number of dashes: %d" % len(dashes))
        print("| Avg. dash length: %s ms" % dashavg)
        print("| Dash stdev: %s ms" % dashsd)
        print("|")
        print("| Number of intra-letter spaces %d" % len(svoids))
        print("| Avg. intra-letter spacing length: %s ms" % svoidavg)
        print("| Intra-letter spacing stdev: %s" % svoidsd)
        print("|")
        print("| Number of inter-letter spaces: %d" % len(mvoids))
        print("| Avg. inter-letter spacing length: %s ms" % mvoidavg)
        print("| Inter-letter spacing stdev: %s ms" % mvoidsd)
        print("|")
        print("| Number of inter-word spaces: %d" % len(lvoids))
        print("| Avg. Inter-word spacing length: %s ms" % lvoidavg)
        print("| Inter-word spacing stdev: %s ms" % lvoidsd)
        print("|")
        print("| Signal duration: %s sec." % duration)
        print("| Approx. elements (dots, dashes) per minute: %s" % epm)
        print("----------")

    if options.statfile:
        try:
            f = open(statfile, "w")
        except:
            print("** Could not open %s for writing")
            exit()

        statstr = "%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f"\
            % (dotavg, dotsd, dashavg, dashsd, svoidavg, svoidsd, mvoidavg, mvoidsd, lvoidavg, lvoidsd)
        f.write(statstr)
        f.close()
        print("** Wrote statistics to %s" % statfile)




    message = morse_to_english(code)

    if options.show_morse:
        sys.stdout.write('** Morse: ')
        for i in range(len(code)):
            if code[i] == '0':
                pass
            elif code[i] == '00':
                sys.stdout.write(' ')
            elif code[i] == '000':
                sys.stdout.write(' [word gap] ')
            else:
                sys.stdout.write(code[i])

        print("")

    print("")
    print("@@ Message: %s" % message)

    if options.code_key:
        decoded = decode(symbols, options.code_key, statstr, code)
        print("@@ Decoded: %s" % decoded)
    print("")

    print("** Finished")
    print("")

#
# Input a list of times
# Return average and SD
#
def morsestats(input):
    if not input:
        return float(0), float(0)
    avg = np.average(input)
    sd = np.std(input)

    return avg, sd

#
# Input a cover message modulated with our transmitter
# Return the covert message
#
def decode(symbols, key, statstr, code):
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
    for c in code:    
        if c == '.':
            avg = dotavg
            sd = dotsd

        elif c == '-':
            avg = dashavg
            sd = dashsd

        elif c == '0':
            avg = inter_void_avg
            sd = inter_void_sd

        elif c == "00":
            avg = letter_void_avg
            sd = letter_void_sd

        elif c == "000":
            avg = word_void_avg
            sd = word_void_sd


        # In the transmitter's randoms array, every 00 is preceded by a 0,
        # and every 000 is preceded by a 00, so compensate for this here
        if sd != 0:
            if c == "000":
                tmpavg = letter_void_avg
                tmpsd = letter_void_sd
                wait = (np.random.normal(tmpavg, tmpsd) % (2*tmpsd)) + (tmpavg-tmpsd)
                both = 1

            if c == "00" or both == 1:
                tmpavg = inter_void_avg
                tmpsd = inter_void_sd
                wait = (np.random.normal(tmpavg, tmpsd) % (2*tmpsd)) + (tmpavg-tmpsd)
                both = 0

            wait = (np.random.normal(avg, sd) % (2*sd)) + (avg-sd)
            # only need the symbol randoms and the letter space
            if c == '.' or c == '-':
                randoms.append(wait)

    outmsg = list()
    for i in range(len(symbols)):
        if abs(randoms[i] - symbols[i]) > (1.5 * sd): # word seperation
            outmsg.append("000")

        elif abs(randoms[i] - symbols[i]) < sd/2: # letter seperation
            outmsg.append("00")

        elif randoms[i] > symbols[i]:
            outmsg.append('-')

        else:
            outmsg.append('.')

    letter = list()
    decoded = list()
    i = 0
    while i < len(outmsg):
        if outmsg[i] == '00' or outmsg[i] == '000':
            try:
                alpha = morse[''.join(letter)]
            except:
                alpha = ' '

            decoded.append(alpha)

            if outmsg[i] == '000':
                decoded.append(' ')

            letter[:] = []
            i = i + 1
            continue

        letter.append(outmsg[i])
        i = i + 1

    decoded = ''.join(decoded)

    # remove trailing '?'s (not applicable now)
    i = len(decoded)-1
    while i != 0:
        if decoded[i] == '?':
            decoded = decoded[0:i]
            i = i - 1
        else:
            break

    return decoded

#
# Take a signal (from a wav file)
# Return a list of ones and zeros
# Ones where the abs of signal is above the threshold
#
def signal_to_rects(signal, tolerance, fs, fc, window_alts):
    winsz =  window_alts * int( math.ceil(fs/fc) ) # window_alts alternations of the signal
    winnum = int( math.floor(len(signal)/winsz) )

    print("** Signal alternations per window: %s" % window_alts)

    rects = list()
    for i in range(winnum):
        num = 0
        samples = list()
        for m in range(winsz):
            try: # fails if a remainder, at the end
                samples.append(abs(signal[(i*winsz) + m]))
            except:
                pass

        if np.average(samples) > tolerance:
            num = 1
            samples[:] = []
            #if(abs(signal[(i * winsz) + m]) >= tolerance):
            #    num = 1
            #    break

        for k in range(winsz):
            try: # because fails if len(signal) % winsz, at the end
                rects.append(num)
            except:
                pass

    #print_rects(rects)
    #exit()
    return rects

#
# Take a list of ones and zeros
# Return a list of symbol times
#
def rects_to_symbols(rects,fs):
    count = 0
    symbols = list()
    voids = list()
    for i in range(len(rects)):
        if rects[i] == 1:
            if rects[i-1] == 0:
                voids.append(count*(1/fs)*1000)
                count = 1
            else:
                count = count + 1
        else:
            if rects[i-1] == 1:
                symbols.append(count*(1/fs)*1000)
                count = 1
            else:
                count = count + 1

    return symbols, voids

#
# Input a list of message times
# Output dots and dashes in seperate lists
#
def symbols_to_beeps(symbols):
    ref = 0
    for i in range(len(symbols)-1):
        if int(symbols[i]) != 0 and symbols[i+1]/symbols[i] > 2:
            ref = symbols[i]
            break
        elif int(symbols[i+1]) != 0 and symbols[i]/symbols[i+1] > 2:
            ref = symbols[i+1]
            break

    dots = list()
    dashes = list()
    for s in symbols:
        if not s > 2 * ref:
            dots.append(s)
        else:
            dashes.append(s)
    
    return dots, dashes

#
# Input a list of times
# Output dots, dashes, and various voids
#
def symbols_to_morse(symbols, voids, dotavg):
    morse = list()
    for i in range(0, len(symbols)):
        if symbols[i] < 2*dotavg:
            morse.append('.')
        else:
            morse.append('-')

        try:
            if voids[i] < 2 * dotavg:
                morse.append('0')
            elif voids[i] < 5 * dotavg:
                morse.append('00')
            else:
                morse.append('000')
        except: # fails at the end b/c we stripped away beginning and trailing voids
            pass
    return morse

#
# Input a list of dots, dashes, and various voids
# Output a message
#
def morse_to_english(morse_input):
    letter = list()
    message = list()
    i = 0
    while i < len(morse_input):
        if morse_input[i] == '0':
            i = i + 1

        elif morse_input[i] == '00':
            # letter seperation
            try:
                alpha = morse[''.join(letter)]
            except:
                alpha = '?'

            message.append(alpha)
            letter[:] = []
            i = i + 1

        elif morse_input[i] == '000':
            # word seperation
            try:
                alpha = morse[''.join(letter)]
            except:
                alpha = '?'

            message.append(alpha)
            message.append(' ')
            letter[:] = []
            i = i + 1
        else:
            letter.append(morse_input[i])
            i = i + 1

    # print out the final letter
    try:
        alpha = morse[''.join(letter)]
    except:
        alpha = '?'
    
    message.append(alpha)

    return ''.join(message)

######################
# Troubleshooting functions
######################

#
# Input: rectangles list
# Output: human-readable rectangles list
#
def print_rects(rects):
    count = 0
    ones = 0
    symbols = 0
    results = list()
    for i in range(len(rects)):
        if rects[i] == 1:
            if ones == 0:
                string = "0:"+str(count)
                results.append(string)
                count = 0
                ones = 1
                symbols = symbols + 1
            count = count + 1
        else:
            if ones == 1:
                string = "1:"+str(count)
                results.append(string)
                count = 0
                ones = 0
            count = count + 1

    print(results)
    print("")
    print("symbols in rects: %s" % symbols)


#########
main()

