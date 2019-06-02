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

class PretrainedHMM():

    def __init__(self,
                 trans_probs: TextIO,
                 emi_probs: TextIO):

        with open(trans_probs) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            header = next(reader, None)  # skip the header
            """
            Our transpositions are stored in a two-dimensional dictionary which is in the following format:
            {'BOS': {'O': BOS-O-prob, 'art': BOS-art-prob, ..., 'tim': BOS-tim-prob}, 'O': {'O': O-O-prob, 'art': O-art-
            prob, ... 'tim': O-tim-prob}, ..., 'tim': {'O': tim-O-prob, 'art': tim-art-prob, ..., 'tim': tim-tim-prob}}
            to be called as: trans_probs[predecessor-pt][successor-pt]
            """
            trans_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
            for row in reader:
                # header contains predecessor pos-tags, first element per row are successor pos-tags
                for i in range(1,len(row)):
                    pos_tag_succ = row[0]
                    pos_tag_pred = header[i]
                    # we calculate in the logarithmic room to base 2
                    trans_probs[pos_tag_pred][pos_tag_succ] = math.log(float(row[i]), 2.0)
            self.trans_probs = trans_probs

        with open(emi_probs) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            """
            order of pos-tag probs per token (self.pos_tags): 
            | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
            | O	|art|eve|geo|gpe|nat|org|per|tim|
            """
            self.pos_tags = next(reader, None)  # skip the header
            del(self.pos_tags[0]) # remove first element, which is 'head'
            self.emi_probs = defaultdict((lambda: [0 for i in range(1, len(self.pos_tags))])) # skip first element, which is key
            for row in reader:
                row_log = []
                token = row[0]
                for i in range(1,len(row)):
                    # we calculate in the logarithmic room to base 2
                    row_log.append(math.log(float(row[i]), 2.0))
                self.emi_probs[token] = row_log

    def tag_sent(self,
                 tok_sent: str): #-> Tuple[List[str], float]:
        sent_lst = []
        for token in tok_sent.rstrip('\n').split(' '):
            sent_lst.append(token)
        viterbi_lst = [[math.log(1.0) for i in range(0, len(self.pos_tags))]
                       for j in range(0, len(sent_lst))]
        # first token can be calculated based on V(0) = 1.0, this is won't be changed again.
        for i in range(0, len(self.pos_tags)):
            emi = float(self.emi_probs[sent_lst[0]][i])
            """
            viterbi_lst[i][j], i = 0-n, j = 0-9:
                 | O |art|eve|geo|gpe|nat|org|per|tim|BOS|
            tok1  0/0 0/1   ...                       0/j
            tok2  1/0 1/1   ...                       1/j
            tok3  
            ...
            toki  n/0 n/1   ...                       n/j
            """
            viterbi_lst[0][i] = math.log(1.0) + float(self.trans_probs['BOS'][self.pos_tags[i]]) + emi
        # loop on token-level
        for i in range(0, len(sent_lst)-1):
            # loop on pos-tag level
            for j in range(0, len(self.pos_tags)):
                transp_vit = []
                # The index of the highest result will have the
                # for every pos-tag of token(i+1) we calculate the maximum out of every transposition with the i-th
                # pos-tag in combination with the respective viterbi-prob(i)
                for next_pt in self.trans_probs[self.pos_tags[j]]:
                    # transp_vit buffers the calculated transpositions between j-th pos-tag of token(i) and every pos-
                    # tag of token(i+1) with viterbi-prob for the j-th pos-tag for token(i) and the emission-prob for
                    # the new token(i+1).
                    transp_vit.append(float(self.trans_probs[self.pos_tags[j]][next_pt]) +
                                      float(viterbi_lst[i][j]))
                # we get the index of the highest calculated probability for every j-th pos-tag at token(i+1), ...
                max_inx_transp_vit = np.argmax(transp_vit)
                # ... calculate the emi_prob for token(i+1) in the sentence ...
                emi = float(self.emi_probs[sent_lst[i + 1]][j])
                #... add the probabilities and store the result in our matrix at the j-th position, which is the resp.
                # pos-tag-column.
                viterbi_lst[i + 1][j] = transp_vit[max_inx_transp_vit] + emi
        # out-list contains the pos-tag sequence with the highest viterbi-probability for every part-sequence
        out_list = []
        # looping through matrix, determining for each token the pos-tag with the highest viterby-prob
        for i in range (0, len(sent_lst)):
            max_indx = np.argmax(viterbi_lst[i])
            out_list.append(self.pos_tags[max_indx])
            viterbi_prob = viterbi_lst[i][max_indx]
        out_tup = (out_list, viterbi_prob)
        return out_tup

"""
Implementation for CLI
"""
@click.command()
@click.option("-t", prompt="Please enter option -t", type=click.Path(),
              help="File containing pretrained transition probabilities.")
@click.option("-e", prompt="Please enter option -e", type=click.Path(),
              help="File containing pretrained emission probabilities.")
@click.option("-o", type=click.Path(),
              help="File to write tagged sentences into.")
@click.option("-i", type=click.Path(),
              help="File containing tokenized sentences.")

def main(t, e, o, i):
    infile_emission_probs = e
    infile_transition_probs = t
    #infile_emission_probs = 'emission_probs.tsv'
    #infile_transition_probs = 'transition_probs.tsv'
    print('\n')
    my_pretrained_hmm = PretrainedHMM(infile_transition_probs, infile_emission_probs)
    #with open('test_sents.txt', 'r', encoding='UTF8') as data:
    #i = 'test_sents.txt'
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