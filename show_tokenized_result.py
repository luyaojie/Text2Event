from transformers import AutoTokenizer
from extraction.event_schema import EventSchema
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='t5-base')
    parser.add_argument('-d', '--data',
                        default='data/text2tree/dyiepp_ace2005_subtype')
    options = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(options.model, use_fast=False)

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<extra_id_0>", "<extra_id_1>"]}
    )

    print(tokenizer)
    print(type(tokenizer))
    print("================================")

    folder_path = options.data

    schema_file = folder_path + "/event.schema"

    event_schema = EventSchema.read_from_file(schema_file)

    subtoken_list = list()
    for typename in event_schema.type_list:
        typename = typename.replace('_', ' ')
        after_tokenzied = tokenizer.encode(typename, add_special_tokens=False)
        subtoken_list += after_tokenzied
        print(typename, after_tokenzied,
              tokenizer.convert_ids_to_tokens(after_tokenzied))
        print("----------------")

    print("================================")

    for instance in open(folder_path + "/val.json").readlines()[:10]:
        instance = json.loads(instance)
        print(tokenizer.tokenize(instance['event']))
        print("----------------")

    for name in ['<extra_id_0>', '<extra_id_1>']:
        print(name, tokenizer.encode(name), tokenizer.tokenize(name))


if __name__ == "__main__":
    main()
