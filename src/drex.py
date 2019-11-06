#!/usr/bin/python
# -*- coding: utf-8 -*-


'''

File: drex.py

DREx expansion program.

Author: Marcelo Pita
Created: 2015/03/17

Modified: 2016/07/16 (Marcelo Pita) (First version)

'''


import sys
import os
import argparse
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import threading
import time
import random
import copy


'''
Represents a document to be expanded.
'''
class Document(object):

    def __init__(self, doc_str):
        doc_values = doc_str.split(';')
        self.doc_id = doc_values[0]
        self.doc_class = doc_values[1]
        self.doc_text = doc_values[2]
        self.doc_expanded_text = None

    def reset_doc_text(self, text):
        self.doc_text = text


'''
'''
class GlobalState(object):

    def __init__(self, m, w, o, lc, lo, c, n, ngram_len):
        self.target_doc_size = m
        self.cache = c
        self.words_model = w
        self.out_file = o
        self.cache_lock = lc
        self.output_lock = lo
        self.num_cache_hits = 0
        self.num_cache_misses = 0
        self.num_not_expanded_docs = 0
        self.num_expansion_words = n
        self.ngram_len = ngram_len
        self.first_output_writing = True

    def get_num_not_expanded_docs(self):
        return self.num_not_expanded_docs
        
    def get_cache(self):
        return self.cache

    def get_num_cache_hits(self):
        return self.num_cache_hits

    def get_num_cache_misses(self):
        return self.num_cache_misses

    def get_target_doc_size(self):
        return self.target_doc_size

    def get_ngram_len(self):
        return self.ngram_len

    def get_ngram_expansion(self, ngram_words):
        ngram_words.sort()
        ngram_str = ngram_words[0]
        if len(ngram_words) > 1:
            for w in ngram_words[1:]:
                ngram_str += '+' + w

        ngram_expansion = {}

        if self.cache.has_key(ngram_str):
            ngram_expansion = self.cache[ngram_str]
            self.num_cache_hits += 1
        else:
            self.num_cache_misses += 1
            if self.words_model is not None:
                try:
                    for k, v in self.words_model.most_similar(positive=ngram_words, topn=self.num_expansion_words):
                        ngram_expansion[k] = v
                except:
                    return None
                self.cache_lock.acquire()
                self.cache[ngram_str] = ngram_expansion
                self.cache_lock.release()
            
        return ngram_expansion

    def insert_word(self, words, word):
        best_pos = 0
        best_sim = float("-inf")
        for i in range(0, len(words)):
            try:
                sim = self.words_model.similarity(word, words[i])
            except:
                continue
            if sim > best_sim:
                best_pos = i + 1
                best_sim = sim
        words.insert(best_pos, word)

    def write_expanded_line(self, doc):
        self.output_lock.acquire()
        try:
            if self.first_output_writing:
                self.first_output_writing = False
            else:
                self.out_file.write('\n')
            self.out_file.write(doc.doc_id + ';' + doc.doc_class + ';' + doc.doc_expanded_text)
            self.out_file.flush()
        finally:
            self.output_lock.release()


def select_word_expansion(ws):
    if not ws:
        return None
    max_val = sum(ws.values())
    pick = random.uniform(0, max_val)
    current = 0
    for key, value in ws.items():
        current += value
        if current > pick:
            return key

def expand_drex_aux(doc, global_state):
    target_doc_size = global_state.get_target_doc_size()
    d = doc
    while True:
        d = expand_drex(d, global_state)
        dlen = len(d.doc_expanded_text.split())
        if dlen >= target_doc_size:
            break
        if dlen == len(d.doc_text.split()):
            global_state.num_not_expanded_docs += 1
            break
        d.doc_text = d.doc_expanded_text
        d.doc_expanded_text = None
    global_state.write_expanded_line(d)
        
def expand_drex(doc, global_state):
    words = doc.doc_text.split()
    words_len = len(words)

    # Saves original document if size is big enough
    target_doc_size = global_state.get_target_doc_size()
    if words_len >= target_doc_size:
        doc.doc_expanded_text = doc.doc_text
        return doc

    # Words for the new doc. Initially all original words.
    # Order is important.
    new_words = list(words)

    ngram_len = global_state.get_ngram_len()

    # Adjust n-gram lenght if too big
    # In this case, the doc itself will be the n-gram
    if ngram_len > words_len:
        n_gram_len = words_len

    # All n-grams
    all_ngrams = zip(*[words[i:] for i in range(ngram_len)])

    # Build "graph" of words similarity
    expansions = {}
    for ngram in all_ngrams:
        ngram_expansion = global_state.get_ngram_expansion(list(ngram))
        if ngram_expansion is None:
            continue
        for potential_word, similarity in ngram_expansion.iteritems():
            if potential_word in expansions:
                expansions[potential_word] += similarity
                continue
            expansions[potential_word] = similarity

    # Insert words until reaching target doc size
    for i in range(0, target_doc_size - words_len):
        selected_word = select_word_expansion(expansions)
        if not selected_word:
            break
        global_state.insert_word(new_words, selected_word)
        expansions.pop(selected_word, None)

    # Expanded text
    try:
        expanded_text = " ".join(new_words)
    except:
        print new_words
        sys.exit()

    doc.doc_expanded_text = expanded_text
    return doc


def main(argv=None):
    sys.setrecursionlimit(1500)

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        description="DREx expansion program.")

    parser.add_argument("-i", "--input", help="Input dataset file", required=True)
    parser.add_argument("-o", "--output", help="Output expanded dataset file", required=True)
    parser.add_argument("-v", "--vectors", help="Words vectors file (word2vec format)")
    parser.add_argument("-b", "--binary", help="Set words vectors file to binary", action='store_true',
                        default=False)
    parser.add_argument("-t", "--target_doc_size", help="Target document size", type=int, default=30)
    parser.add_argument("-l", "--ngram_len", help="N-gram length", type=int, default=2)
    parser.add_argument("-n", "--num_expansion_words", help="Number of expansion words per ngram",
                        type=int, default=100)
    parser.add_argument("-c", "--cache_filename", help="Cache file")
    parser.add_argument("-p", "--processing_threads", help="Number of processing threads",
                        type=int, default=4)

    args = parser.parse_args(argv[1:])

    num_lines = 0
    in_file = open(args.input, "r")
    for line in in_file:
        num_lines += 1
    in_file.close()
    print "Number of lines: " + str(num_lines)

    in_file = open(args.input, "r")
    out_file = open(args.output, "w")

    print "Number of threads: " + str(args.processing_threads)
    sys.stdout.flush()

    print "Loading cache file:",
    cache = {}
    sys.stdout.flush()
    cache_file = None
    try:
        cache_file = open(args.cache_filename, "r")
        i = 0
        for e in cache_file.read().splitlines():
            words = e.split()
            if len(words) < 1:
                continue
            i += 1
            kw = words[0].strip()
            exps = {}
            if len(words) > 1:
                for w in words[1:]:
                    word_and_sim = w.split(',')
                    exps[word_and_sim[0]] = float(word_and_sim[1])
            cache[kw] = exps
        cache_file.close()
        print str(i) + " entries found"
        sys.stdout.flush()
    except Exception, e:
        print str(e)
        sys.stdout.flush()

    words_model = None
    if args.vectors is not None:
        try:
            print "Loading words vectors model...",
            sys.stdout.flush()
            words_model = None
            if args.binary:
		words_model = KeyedVectors.load_word2vec_format(args.vectors, binary=True)
            else:
		words_model = KeyedVectors.load_word2vec_format(args.vectors, binary=False)
            words_model.init_sims(replace=True)
        except:
            print "FAILED!"
            sys.stdout.flush()
        print "OK!"
        sys.stdout.flush()

    cache_lock = threading.Lock()
    output_lock = threading.Lock()

    global_state = GlobalState(args.target_doc_size, words_model, out_file, cache_lock, output_lock, cache, args.num_expansion_words,
                               args.ngram_len)

    max_num_active_threads = 0

    for i in range(1, num_lines+1):
        num_active_threads = threading.activeCount()

        if (num_active_threads - 1) > max_num_active_threads:
            max_num_active_threads = num_active_threads - 1

        if ((i-1) % 10 == 0):
            sys.stdout.write("Running (count: %d / %d) (threads: %d active) (cache: %d [h] | %d [m]) (incomp: %d)...\r" % (i, num_lines, (num_active_threads-1), global_state.get_num_cache_hits(), global_state.get_num_cache_misses(), global_state.get_num_not_expanded_docs()  ) )
            sys.stdout.flush()

        doc = Document(in_file.readline())

        while (num_active_threads >= args.processing_threads+1):
            time.sleep(1)
            num_active_threads = threading.activeCount()

        t = threading.Thread(target=expand_drex_aux, args=(doc, global_state,))
        t.start()

    num_active_threads = threading.activeCount() - 1

    if num_active_threads > max_num_active_threads:
        max_num_active_threads = num_active_threads

    sys.stdout.write("Running (count: %d / %d) (threads: %d active) (cache: %d [h] | %d [m]) (incomp: %d)... OK!\n" % (i, num_lines, num_active_threads, global_state.get_num_cache_hits(), global_state.get_num_cache_misses(), global_state.get_num_not_expanded_docs()  ) )
    sys.stdout.flush()

    in_file.close()

    while (num_active_threads > 0):
        time.sleep(1)
        num_active_threads = threading.activeCount() - 1
        sys.stdout.write("Remaning threads: %d         \r" % (num_active_threads,))
        sys.stdout.flush()
    out_file.close()

    cache = global_state.get_cache()

    if args.cache_filename is not None:
        try:
            print "Writing cache file:",
            sys.stdout.flush()
            cache_file = open(args.cache_filename, "w")
            for k,v in cache.iteritems():
                v_str = []
                for ek,ev in v.iteritems():
                    v_str.append(ek + "," + str(ev))
                cache_file.write(k + " " + " ".join(v_str) + "\n")
            cache_file.close()
            print str(len(cache)) + " entries"
            sys.stdout.flush()
        except:
            print "FAILED!"
            sys.stdout.flush()


if __name__ == "__main__":
    main()
