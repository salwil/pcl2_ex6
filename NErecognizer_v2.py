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
import sys

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
            del(self.pos_tags[0]) # remove first element, which is 'head'
            self.emi_probs = defaultdict((lambda: [0 for i in range(1, len(self.pos_tags))])) # skip first element, which is key
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
                self.emi_probs[token] = row_log
            print('German', self.emi_probs['German'])
            print(self.pos_tags)

    def tag_sent(self,
                 tok_sent: str): #-> Tuple[List[str], float]:
        sent_lst = []
        for token in tok_sent.rstrip('\n').split(' '):
            sent_lst.append(token)
        prev_pos_tag = 'BOS'
        viterbi = math.log(1.0)
        viterbi_lst = [[math.log(1.0) for i in range(0, len(self.pos_tags))]
                       for j in range(0, len(sent_lst))]
        # first token can be calculated based on V(0) = 1.0, this is final
        for i in range(0, len(self.pos_tags)):
            emi = float(self.emi_probs[sent_lst[0]][i])
            viterbi_lst[0][i] = math.log(1.0) + float(self.trans_probs['BOS'][self.pos_tags[i]]) + emi
        for i in range(0, len(sent_lst)-1):
            # we calculate for every
            for j in range(0, len(self.pos_tags)):
                # we're already calculating the emi_prob for the next token in the sentence
                emi = float(self.emi_probs[sent_lst[i + 1]][j])
                transp_vit = []
                # The index of the highest result will have the
                k = 0
                for next_pt in self.trans_probs[self.pos_tags[j]]:
                    transp_vit.append(float(self.trans_probs[self.pos_tags[j]][next_pt]) +
                                      float(viterbi_lst[i][j]) + emi)
                    k += 1
                max_inx_transp_vit = np.argmax(transp_vit)
                prev_pt_max = self.pos_tags[max_inx_transp_vit-1]
                #viterbi_lst[i+1][j] = transp_vit[max_inx_transp_vit] + emi
                viterbi_lst[i + 1][j] = transp_vit[max_inx_transp_vit]
        out_list = ['BOS']
        for i in range (0, len(sent_lst)):
            max_indx = np.argmax(viterbi_lst[i])
            out_list.append(self.pos_tags[max_indx])
            viterbi_prob = viterbi_lst[i][max_indx]
        del(out_list[0])
        out_tup = (out_list, viterbi_prob)
        print(out_tup)
        return out_tup



"""
Implementation for CLI
"""
@click.command()
#@click.option("-t", prompt="Please enter option - t", type=click.Path(),
#              help="File containing pretrained transition probabilities.")
#@click.option("-e", prompt="Please enter option - e", type=click.Path(),
#              help="File containing pretrained emission probabilities.")
@click.option("-o", type=click.Path(),
              help="File to write tagged sentences into.")
@click.option("-i", type=click.Path(),
              help="File containing tokenized sentences.")

def main(o, i):
    #infile_emission_probs = e
    #infile_transition_probs = t
    infile_emission_probs = 'emission_probs.tsv'
    infile_transition_probs = 'transition_probs.tsv'
    print('\n')
    my_pretrained_hmm = PretrainedHMM(infile_transition_probs, infile_emission_probs)
    #with open('test_sents.txt', 'r', encoding='UTF8') as data:
    i = 'test_sents.txt'
    if o:
        outfile = open(o, "w", encoding='UTF8')
    if i:
        test_file = open(i, "r", encoding='UTF8')
        for line in test_file:
            token_lst, calculated_probability = my_pretrained_hmm.tag_sent(line)
            out_prob = str(calculated_probability)
            outline_1 = ['Sequence Probability: ', out_prob, '\n']
            j = 0
            outline_2 = []
            for token in line.rstrip('\n').split(' '):
                outline_2.append(token)
                outline_2.append('_')
                outline_2.append(token_lst[j])
                outline_2.append(' ')
                j+=1
            outline_2.append('\n')
            if o:
                outfile.write("".join(outline_1))
                outfile.write("".join(outline_2))
            else:
                print("".join(outline_1))
                print("".join(outline_2))
        test_file.close()
    else:
        for line in sys.stdin:
            test_file = line
            if test_file > ' ':
                pass
            else:
                print('Missing argument')
                break
        token_lst, calculated_probability = my_pretrained_hmm.tag_sent(test_file)
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