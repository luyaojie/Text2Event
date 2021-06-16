#!/usr/bin/env python
# -*- coding:utf-8 -*-
from copy import deepcopy
from typing import List, Counter, Tuple

EVENT_EXTRACTION_KEYS = ["trigger-P", "trigger-R", "trigger-F1",
                         "role-P", "role-R", "role-F1"]


class PredictParser:
    def __init__(self, label_constraint):
        self.predicate_set = label_constraint.type_list
        self.role_set = label_constraint.role_list

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List, Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            Counter:
        """
        pass

    @staticmethod
    def count_multi_event_role_in_instance(instance, counter):
        if len(instance['gold_event']) != len(set(instance['gold_event'])):
            counter.update(['multi-same-event-gold'])

        if len(instance['gold_role']) != len(set(instance['gold_role'])):
            counter.update(['multi-same-role-gold'])

        if len(instance['pred_event']) != len(set(instance['pred_event'])):
            counter.update(['multi-same-event-pred'])

        if len(instance['pred_role']) != len(set(instance['pred_role'])):
            counter.update(['multi-same-role-pred'])


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
