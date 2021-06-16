#!/usr/bin/env python
# -*- coding:utf-8 -*-
from data_convert.format.target_format import TargetFormat


def get_str_from_tokens(tokens, sentence, separator=' '):
    start, end_exclude = tokens[0], tokens[-1] + 1
    return separator.join(sentence[start:end_exclude])


type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
role_start = '<extra_id_2>'
role_end = '<extra_id_3>'


class Text2Tree(TargetFormat):

    @staticmethod
    def annotate_predicate_arguments(tokens, predicate_arguments, mark_tree=False, multi_tree=False, zh=False):
        """

        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :param mark_tree False
            (
                (Meet summit (Entity Russia) (Entity major industrialized nations))
                (Phone-Write told (Entity US President George W. Bush) (Entity Canadian Prime Minister Jean Chretien) (Time-Within Monday))
            )
       :param  multi_tree  True
            (Meet summit [Entity Russia] [Entity major industrialized nations])
            (Phone-Write told [Entity US President George W. Bush] [Entity Canadian Prime Minister Jean Chretien] [Time-Within Monday])

        :return:
        """
        token_separator = '' if zh else ' '

        event_str_rep_list = list()

        for predicate_argument in predicate_arguments:
            event_type = predicate_argument['type']

            predicate_text = get_str_from_tokens(predicate_argument['tokens'], tokens, separator=token_separator)

            # prefix_tokens[predicate_argument['tokens'][0]] = ['[ ']
            # suffix_tokens[predicate_argument['tokens'][-1]] = [' ]']

            role_str_list = list()
            for role_name, role_tokens in predicate_argument['arguments']:
                # if role_name == 'Place' or role_name.startswith('Time'):
                if role_name == event_type:
                    continue

                role_text = get_str_from_tokens(role_tokens, tokens, separator=token_separator)
                if mark_tree:
                    role_str = ' '.join([role_start, role_name, role_text, role_end])
                else:
                    role_str = ' '.join([type_start, role_name, role_text, type_end])
                role_str_list += [role_str]
            role_str_list_str = ' '.join(role_str_list)
            event_str_rep = f"{type_start} {event_type} {predicate_text} {role_str_list_str} {type_end}"
            event_str_rep_list += [event_str_rep]

        source_text = token_separator.join(tokens)
        target_text = ' '.join(event_str_rep_list)

        if not multi_tree:
            target_text = f'{type_start} ' + \
                          ' '.join(event_str_rep_list) + f' {type_end}'

        return source_text, target_text

    @staticmethod
    def annotate_span(tokens, predicate_arguments, mark_tree=False, zh=False):
        """

        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :return:
        mark_tree False
            (
                (Meet summit (Entity Russia) (Entity major industrialized nations))
                (Phone-Write told (Entity US President George W. Bush) (Entity Canadian Prime Minister Jean Chretien) (Time-Within Monday))
            )
        mark_tree  True
            (
                (Meet summit [Entity Russia] [Entity major industrialized nations])
                (Phone-Write told [Entity US President George W. Bush] [Entity Canadian Prime Minister Jean Chretien] [Time-Within Monday])
            )
        """

        token_separator = '' if zh else ' '

        event_str_rep_list = list()

        for predicate_argument in predicate_arguments:
            event_type = predicate_argument['type']

            predicate_text = get_str_from_tokens(predicate_argument['tokens'], tokens, separator=token_separator)

            span_str_list = [' '.join([type_start, event_type, predicate_text, type_end])]

            for role_name, role_tokens in predicate_argument['arguments']:
                # if role_name == 'Place' or role_name.startswith('Time'):
                if role_name == event_type:
                    continue

                role_text = get_str_from_tokens(role_tokens, tokens, separator=token_separator)
                if mark_tree:
                    role_str = ' '.join([role_start, role_name, role_text, role_end])
                else:
                    role_str = ' '.join([type_start, role_name, role_text, type_end])
                span_str_list += [role_str]
            event_str_rep_list += [' '.join(span_str_list)]

        source_text = token_separator.join(tokens)
        target_text = f'{type_start} ' + ' '.join(event_str_rep_list) + f' {type_end}'

        return source_text, target_text


if __name__ == "__main__":
    pass
