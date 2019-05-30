#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Roger Rüttimann rroger 02-914-471
# Salome Wildermuth salomew 10-289-544

from collections import defaultdict
from typing import TextIO
import numpy as np
import csv
import math
import click

"""
Fragen: Wahrscheinlichkeiten im logarithmischen Zahlenraum zurückgeben?
Was soll passieren, wenn der Benutzer die Option -i eingibt?
"""

class PretrainedHMM():

    def __init__(self,
                 trans_probs: TextIO,
                 emi_probs: TextIO):

        with open(trans_probs) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            header = next(reader, None)  # skip the header
            trans_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
            for row in reader:
                """
                order of probs per pos-tag-successor in row_log: 
                | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
                | O	|art|eve|geo|gpe|nat|org|per|tim|BOS|
                """
                for i in range(1,len(row)):
                    pos_tag_succ = row[0]
                    pos_tag_prec = header[i]
                    trans_probs[pos_tag_prec][pos_tag_succ] = (math.log(float(row[i])))
            self.trans_probs = trans_probs

        with open(emi_probs) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            self.pos_tags = next(reader, None)  # skip the header
            emi_probs = defaultdict((lambda: [0 for i in range(1, len(self.pos_tags))])) # skip first element, which is key
            for row in reader:
                """
                order of probs pos-tag per token in row_log: 
                | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
                | O	|art|eve|geo|gpe|nat|org|per|tim|
                """
                row_log = []
                token = row[0]
                for i in range(1,len(row)):
                    row_log.append(math.log(float(row[i])))
                emi_probs[token] = row_log
            self.emi_probs = emi_probs

    def tag_sent(self,
                 tok_sent: str): #-> Tuple[List[str], float]:
        sent_lst = []
        for token in tok_sent.split(' '):
            sent_lst.append(token)
        prev_pos_tag = 'BOS'
        viterbi = 1.0
        out_list = []
        for token in sent_lst:
            viterbi_lst = []
            for i in range(0, len(self.emi_probs[token])-1):
                curr_pos_tag = self.pos_tags[i]
                viterbi_lst.append((float(self.emi_probs[token][i]) +
                                   float(self.trans_probs[curr_pos_tag][prev_pos_tag])) + viterbi)
            max_indx = np.argmax(viterbi_lst)
            viterbi = viterbi_lst[max_indx]
            prev_pos_tag = self.pos_tags[max_indx+1]
            out_list.append(prev_pos_tag)
        out_tup = (out_list, viterbi)
        return out_tup

"""
Implementation for CLI
"""
@click.command()
@click.option("-t", prompt="Please enter option - t", type=click.Path(),
              help="File containing pretrained transition probabilities.")
@click.option("-e", prompt="Please enter option - e", type=click.Path(),
              help="File containing pretrained emission probabilities.")
@click.option("-o", type=click.Path(),
              help="File to write tagged sentences into.")
@click.option("-i", type=click.Path(),
              help="File containing tokenized sentences.")

def main(t, e, o, i):
    infile_emission_probs = e
    infile_transition_probs = t
    print('\n')
    my_pretrained_hmm = PretrainedHMM(infile_transition_probs, infile_emission_probs)
    with open('test_sents.txt', 'r', encoding='UTF8') as data:
        if o:
            outfile = open(o, "w")
        for line in data:
            token_lst, calculated_probability = my_pretrained_hmm.tag_sent(line)
            out_prob = str(calculated_probability)
            outline_1 = ['Sequence Probability: ', out_prob, '\n']
            i = 0
            outline_2 = []
            for token in line.rstrip('\n').split(' '):
                outline_2.append(token)
                outline_2.append('_')
                outline_2.append(token_lst[i])
                outline_2.append(' ')
                i+=1
            outline_2.append('\n')
            if o:
                outfile.write("".join(outline_1))
                outfile.write("".join(outline_2))
            else:
                print("".join(outline_1))
                print("".join(outline_2))
        if o:
            outfile.close()

if __name__ == '__main__':
        main()