#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict

from data_convert.format.text2tree import type_start, type_end
from extraction.label_tree import get_label_name_tree

import os

debug = True if 'DEBUG' in os.environ else False
debug_step = True if 'DEBUG_STEP' in os.environ else False


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


def find_bracket_position(generated_text, _type_start, _type_end):
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position


def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):
    print(generated, src_sequence) if debug else None

    if len(generated) == 0:
        # It has not been generated yet. All SRC are valid.
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    if end_sequence_search_tokens:
        valid_token += end_sequence_search_tokens

    return valid_token


def get_constraint_decoder(tokenizer, type_schema, decoding_schema, source_prefix=None):
    if decoding_schema == 'tree':
        return TreeConstraintDecoder(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
    elif decoding_schema == 'treespan':
        return SpanConstraintDecoder(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
    else:
        raise NotImplementedError(
            'Type Schema %s, Decoding Schema %s do not map to constraint decoder.' % (
                decoding_schema, decoding_schema)
        )


class ConstraintDecoder:
    def __init__(self, tokenizer, source_prefix):
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []

    def get_state_valid_tokens(self, src_sentence: List[str], tgt_generated: List[str]) -> List[str]:
        pass

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        if debug:
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))

        valid_token_ids = self.get_state_valid_tokens(
            src_sentence.tolist(),
            tgt_generated.tolist()
        )

        if debug:
            print('========================================')
            print('valid tokens:', self.tokenizer.convert_ids_to_tokens(
                valid_token_ids), valid_token_ids)
            if debug_step:
                input()

        # return self.tokenizer.convert_tokens_to_ids(valid_tokens)
        return valid_token_ids


class TreeConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_tree = get_label_name_tree(
            type_schema.type_list, self.tokenizer, end_symbol=self.tree_end)
        self.role_tree = get_label_name_tree(
            type_schema.role_list, self.tokenizer, end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_trigger'
        elif start_number == end_number + 3:
            state = 'generate_role'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(
                self.tokenizer.eos_token_id)]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_trigger':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                # EVENT_TYPE_LEFT: Start a new role
                # EVENT_TYPE_RIGHT: End this event
                return [self.type_start, self.type_end]
            else:
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    src_sentence=src_sentence,
                    end_sequence_search_tokens=[self.type_start, self.type_end]
                )

        elif state == 'generate_role':

            if tgt_generated[-1] == self.type_start:
                # Start Role Label
                return list(self.role_tree.keys())

            generated = tgt_generated[index + 1:]
            valid_tokens = self.search_prefix_tree_and_sequence(
                generated=generated,
                prefix_tree=self.role_tree,
                src_sentence=src_sentence,
                end_sequence_search_tokens=[self.type_end]
            )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % self.tokenizer.convert_ids_to_tokens(
            valid_tokens)) if debug else None
        return valid_tokens


class SpanConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_tree = get_label_name_tree(type_schema.type_list + type_schema.role_list,
                                             tokenizer=self.tokenizer,
                                             end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(
                self.tokenizer.eos_token_id)]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' %
                                   (self.type_end, tgt_generated))

            else:
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    src_sentence=src_sentence,
                    end_sequence_search_tokens=[self.type_end]
                )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens
