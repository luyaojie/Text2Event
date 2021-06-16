#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict


class TargetFormat:
    @staticmethod
    def annotate_spans(tokens: List[str], spans: List[Dict]): pass

    @staticmethod
    def annotate_predicate_arguments(tokens: List[str], predicate_arguments: List[Dict], zh=False): pass
