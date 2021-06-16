import argparse
import json
import os
import sys
import numpy as np
from copy import deepcopy
from pprint import pprint
from extraction.event_schema import EventSchema
from extraction.predict_parser.tree_predict_parser import TreePredictParser


def read_file(file_name):
    return [line.strip() for line in open(file_name).readlines()]


def generate_sentence_dyiepp(filename, type_format='subtype'):
    for line in open(filename):
        instance = json.loads(line)
        sentence = instance['sentence']
        sentence_start = instance.get(
            's_start', instance.get('_sentence_start'))
        events = instance['event']

        trigger_list = list()
        role_list = list()

        for event in events:
            trigger, event_type = event[0]
            trigger -= sentence_start

            suptype, subtype = event_type.split('.')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            trigger_list += [(event_type, (trigger, trigger))]
            for start, end, role in event[1:]:
                start -= sentence_start
                end -= sentence_start
                role_list += [(event_type, role, (start, end))]

        yield ' '.join(sentence), trigger_list, role_list


def generate_sentence_oneie(filename, type_format='subtype'):
    for line in open(filename):
        instance = json.loads(line)
        entities = {entity['id']: entity for entity in instance['entity_mentions']}

        trigger_list = list()
        role_list = list()

        for event in instance['event_mentions']:
            suptype, subtype = event['event_type'].split(':')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            trigger_list += [(event_type, (event['trigger']
                                           ['start'], event['trigger']['end'] - 1))]

            for argument in event['arguments']:
                argument_entity = entities[argument['entity_id']]
                role_list += [(event_type, argument['role'],
                               (argument_entity['start'], argument_entity['end'] - 1))]

        yield ' '.join(instance['tokens']), trigger_list, role_list


def match_sublist(the_list, to_match):
    """
    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def record_to_offset(instance):
    """
    Find Role's offset using closest matched with trigger work.
    :param instance:
    :return:
    """
    trigger_list = list()
    role_list = list()

    token_list = instance['text'].split()

    trigger_matched_set = set()
    for record in instance['pred_record']:
        event_type = record['type']
        trigger = record['trigger']
        matched_list = match_sublist(token_list, trigger.split())

        trigger_offset = None
        for matched in matched_list:
            if matched not in trigger_matched_set:
                trigger_list += [(event_type, matched)]
                trigger_offset = matched
                trigger_matched_set.add(matched)
                break

        # No trigger word, skip the record
        if trigger_offset is None:
            break

        for _, role_type, text_str in record['roles']:
            matched_list = match_sublist(token_list, text_str.split())
            if len(matched_list) == 1:
                role_list += [(event_type, role_type, matched_list[0])]
            elif len(matched_list) == 0:
                sys.stderr.write("[Cannot reconstruct]: %s %s\n" %
                                 (text_str, token_list))
            else:
                abs_distances = [abs(match[0] - trigger_offset[0])
                                 for match in matched_list]
                closest_index = np.argmin(abs_distances)
                role_list += [(event_type, role_type,
                               matched_list[closest_index])]

    return instance['text'], trigger_list, role_list


class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list, verbose=False):
        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', dest='gold_folder')
    parser.add_argument('-r', dest='offset_folder')
    parser.add_argument('-p', dest='pred_folder')
    parser.add_argument('-f', dest='format', default="dyiepp")
    parser.add_argument('-v', dest='verbose', action='store_true')
    options = parser.parse_args()

    if options.format == 'dyiepp':
        data_dict = {
            'valid': ['eval_preds_seq2seq.txt', 'val.json', "dev_convert.json"],
            'test': ['test_preds_seq2seq.txt', 'test.json', "test_convert.json"],
        }
        generate_sentence = generate_sentence_dyiepp
    elif options.format == 'oneie':
        data_dict = {
            'valid': ['eval_preds_seq2seq.txt', 'val.json', "dev.oneie.json"],
            'test': ['test_preds_seq2seq.txt', 'test.json', "test.oneie.json"],
        }
        generate_sentence = generate_sentence_oneie
    else:
        raise NotImplementedError('%s not support' % options.format)

    label_schema = EventSchema.read_from_file(
        filename=os.path.join(options.gold_folder, 'event.schema')
    )

    pred_folder = options.pred_folder
    gold_folder = options.gold_folder
    offset_folder = options.offset_folder

    for data_key, (generation, text_file, offset_file) in data_dict.items():

        trigger_metric = Metric()
        argument_metric = Metric()

        # Reconstruct the offset of predicted event records.
        text_filename = os.path.join(gold_folder, text_file)
        pred_filename = os.path.join(pred_folder, generation)

        print("pred:", pred_filename)
        pred_reader = TreePredictParser(label_constraint=label_schema)
        event_list, _ = pred_reader.decode(
            gold_list=[],
            pred_list=read_file(pred_filename),
            text_list=[json.loads(line)['text']
                       for line in read_file(text_filename)],
        )
        pred_list = [
            record_to_offset(decoding_instance) for decoding_instance in event_list
        ]

        # Read gold event annotation with offsets.
        gold_filename = os.path.join(offset_folder, offset_file)
        print("gold:", gold_filename)
        gold_list = [event for event in generate_sentence(gold_filename)]

        for pred, gold in zip(pred_list, gold_list):
            assert pred[0] == gold[0]
            trigger_metric.count_instance(
                gold_list=gold[1],
                pred_list=pred[1],
                verbose=options.verbose,
            )
            argument_metric.count_instance(
                gold_list=gold[2],
                pred_list=pred[2],
                verbose=options.verbose,
            )

        trigger_result = trigger_metric.compute_f1(prefix=data_key + '-trig-')
        role_result = argument_metric.compute_f1(prefix=data_key + '-role-')

        pprint(trigger_result)
        pprint(role_result)


if __name__ == "__main__":
    main()
