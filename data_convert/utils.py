#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from collections import defaultdict

from tabulate import tabulate


def read_file(filename):
    from tqdm import tqdm
    num_lines = sum(1 for _ in open(filename, 'r'))
    with open(filename, 'r') as f:
        for line in tqdm(f, total=num_lines):
            yield line


def check_output(filename, line_num=2):
    import os
    os.system('tail -n %s %s*' % (line_num, filename))


def data_counter_to_table(data_counter):
    table = list()
    for filename, file_counter in data_counter.items():
        table += [[filename, file_counter['sentence'],
                   file_counter['event'], file_counter['argument']]]
    return tabulate(table, headers=['file', '#sent', '#event', '#arg'])


def get_schema(event):
    event_type = event['type']
    if len(event['arguments']) == 0:
        return {(event_type, None)}
    return set([(event_type, argument[0]) for argument in event['arguments']])


def output_schema(event_schema_set, output_file):
    event_type_list = list(set([schema[0] for schema in event_schema_set]))
    argument_role_list = list(set([schema[1] for schema in event_schema_set]))

    if None in argument_role_list:
        # Same Event only Type without argument
        argument_role_list.remove(None)

    event_type_set_dict = defaultdict(set)

    for event_type, arg_role in event_schema_set:
        if arg_role is None:
            continue
        event_type_set_dict[event_type].add(arg_role)

    event_type_list_dict = defaultdict(list)

    for event_type in event_type_set_dict:
        event_type_list_dict[event_type] = list(
            event_type_set_dict[event_type])

    with open(output_file, 'w') as output:
        output.write(json.dumps(event_type_list) + '\n')
        output.write(json.dumps(argument_role_list) + '\n')
        output.write(json.dumps(event_type_list_dict) + '\n')
