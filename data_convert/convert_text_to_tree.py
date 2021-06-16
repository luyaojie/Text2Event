#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
from collections import Counter, defaultdict
from data_convert.format.text2tree import Text2Tree
from data_convert.task_format.event_extraction import Event, DyIEPP
from data_convert.utils import read_file, check_output, data_counter_to_table, get_schema, output_schema
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english') + ["'s", "'re", "%"])


def convert_file_tuple(file_tuple, data_class=Event, target_class=Text2Tree,
                       output_folder='data/text2tree/framenet',
                       ignore_nonevent=False, zh=False,
                       mark_tree=False, type_format='subtype'):
    counter = defaultdict(Counter)
    data_counter = defaultdict(Counter)

    event_schema_set = set()

    span_output_folder = output_folder + '_span'

    if not os.path.exists(span_output_folder):
        os.makedirs(span_output_folder)

    for in_filename, output_filename in file_tuple(output_folder):
        span_output_filename = output_filename.replace(
            output_folder, span_output_folder)

        event_output = open(output_filename + '.json', 'w')
        span_event_output = open(span_output_filename + '.json', 'w')

        for line in read_file(in_filename):
            document = data_class(json.loads(line.strip()))
            for sentence in document.generate_sentence(type_format=type_format):

                if ignore_nonevent and len(sentence['events']) == 0:
                    continue

                source, target = target_class.annotate_predicate_arguments(
                    tokens=sentence['tokens'],
                    predicate_arguments=sentence['events'],
                    zh=zh
                )

                for event in sentence['events']:
                    event_schema_set = event_schema_set | get_schema(event)
                    sep = '' if zh else ' '
                    predicate = sep.join([sentence['tokens'][index]
                                          for index in event['tokens']])
                    counter['pred'].update([predicate])
                    counter['type'].update([event['type']])
                    data_counter[in_filename].update(['event'])
                    for argument in event['arguments']:
                        data_counter[in_filename].update(['argument'])
                        counter['role'].update([argument[0]])

                data_counter[in_filename].update(['sentence'])

                event_output.write(json.dumps(
                    {'text': source, 'event': target}, ensure_ascii=False) + '\n')

                span_source, span_target = target_class.annotate_span(
                    tokens=sentence['tokens'],
                    predicate_arguments=sentence['events'],
                    zh=zh,
                    mark_tree=mark_tree
                )

                span_event_output.write(
                    json.dumps({'text': span_source, 'event': span_target}, ensure_ascii=False) + '\n')

        event_output.close()
        span_event_output.close()

        check_output(output_filename)
        check_output(span_output_filename)
        print('\n')
    output_schema(event_schema_set, output_file=os.path.join(
        output_folder, 'event.schema'))
    output_schema(event_schema_set, output_file=os.path.join(
        span_output_folder, 'event.schema'))
    print('Pred:', len(counter['pred']), counter['pred'].most_common(10))
    print('Type:', len(counter['type']), counter['type'].most_common(10))
    print('Role:', len(counter['role']), counter['role'].most_common(10))
    print(data_counter_to_table(data_counter))
    print('\n\n\n')


def convert_ace2005_event(output_folder='data/text2tree/ace2005_event', type_format='subtype',
                          ignore_nonevent=False, mark_tree=False):
    from data_convert.task_format.event_extraction import ace2005_en_file_tuple
    convert_file_tuple(file_tuple=ace2005_en_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       )


def convert_dyiepp_event(output_folder='data/text2tree/ace2005_event', type_format='subtype',
                         ignore_nonevent=False, mark_tree=False):
    from data_convert.task_format.event_extraction import DyIEPP_ace2005_file_tuple
    convert_file_tuple(file_tuple=DyIEPP_ace2005_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       data_class=DyIEPP
                       )


def convert_ere_event(output_folder='data/text2tree/ere_event', type_format='subtype',
                      ignore_nonevent=False, mark_tree=False):
    from data_convert.task_format.event_extraction import ere_en_file_tuple
    convert_file_tuple(file_tuple=ere_en_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       )


if __name__ == "__main__":
    type_format_name = 'subtype'
    convert_dyiepp_event("data/text2tree/dyiepp_ace2005_%s" % type_format_name,
                         type_format=type_format_name,
                         ignore_nonevent=False, mark_tree=False,
                         )
    convert_ace2005_event("data/text2tree/one_ie_ace2005_%s" % type_format_name,
                          type_format=type_format_name,
                          ignore_nonevent=False,
                          mark_tree=False
                          )
    convert_ere_event("data/text2tree/one_ie_ere_en_%s" % type_format_name,
                      type_format=type_format_name,
                      ignore_nonevent=False,
                      mark_tree=False)
