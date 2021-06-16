#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
from os import path
import json
import collections


def main():

    output_dir = sys.argv[1]
    for fold in ["train", "dev", "test"]:
        g_convert = open(path.join(output_dir, fold + "_convert.json"), "w")
        with open(path.join(output_dir, fold + ".json"), "r") as g:
            print('convert %s to %s' % (
                path.join(output_dir, fold + ".json"),
                path.join(output_dir, fold + "_convert.json")
            ))
            for line in g:
                line = json.loads(line)
                sentences = line["sentences"]
                ner = line["ner"]
                relations = line["relations"]
                events = line["events"]
                sentence_start = line["_sentence_start"]
                doc_key = line["doc_key"]

                assert len(sentence_start) == len(ner) == len(
                    relations) == len(events) == len(sentence_start)

                for sentence, ner, relation, event, s_start in zip(sentences, ner, relations, events, sentence_start):
                    sentence_annotated = collections.OrderedDict()
                    sentence_annotated["sentence"] = sentence
                    sentence_annotated["s_start"] = s_start
                    sentence_annotated["ner"] = ner
                    sentence_annotated["relation"] = relation
                    sentence_annotated["event"] = event

                    g_convert.write(json.dumps(
                        sentence_annotated, default=int) + "\n")


if __name__ == "__main__":
    main()
