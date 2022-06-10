# Text2Event

- An implementation for [``Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction``](https://aclanthology.org/2021.acl-long.217)
- Please contact [Yaojie Lu](http://luyaojie.github.io) ([@luyaojie](mailto:yaojie2017@iscas.ac.cn)) for questions and suggestions.

## Update
- [2021-08-03] Update pre-trained models

## Quick links
* [Requirements](#Requirements)
* [Quick Start](#Quick-start)
  * [Data Format](#Data-Format)
  * [Model Training](#Model-Training)
  * [Model Evaluation](#Model-Evaluation)
  * [How to expand to other tasks](#How-to-expand-to-other-tasks)
* [Pre-trained Model](#Pre-trained-Model)
* [Event Datasets Preprocessing](#Event-Datasets-Preprocessing)
* [Citation](#Citation)

## Requirements

General

- Python (verified on 3.8)
- CUDA (verified on 11.1)

Python Packages

- see requirements.txt

```bash
conda create -n text2event python=3.8
conda activate text2event
pip install -r requirements.txt
```

## Quick Start

### Data Format

Data folder contains four files (The detailed preprocessing steps refer to `Event Datasets Preprocessing`):

```text
data/text2tree/one_ie_ace2005_subtype
├── event.schema
├── test.json
├── train.json
└── val.json
```

train/val/test.json are data files, and each line is a JSON instance.
Each JSON instance contains `text` and `event` fields, in which `text` is plain text, and `event` is event linearized form.
If you want to use other key names, it is easy to change the input format in `run_seq2seq.py`.

```text
{"text": "He also owns a television and a radio station and a newspaper .", "event": "<extra_id_0>  <extra_id_1>"}
{"text": "' ' For us the United Natgions is the key authority '' in resolving the Iraq crisis , Fischer told reporters opn arrival at the EU meeting .", "event": "<extra_id_0> <extra_id_0> Meet meeting <extra_id_0> Entity EU <extra_id_1> <extra_id_1> <extra_id_1>"}
```

Note:
- Use the extra character of T5 as the structure indicators, such as `<extra_id_0>`, `<extra_id_1>`, etc.
- `event.schema` is the event schema file for building the trie of constrained decoding.
It contains three lines: the first line is event type name list, the second line is event role name list, the third line is type-to-role dictionary.

  ```text
  ["Declare-Bankruptcy", "Convict", ...]
  ["Plaintiff", "Target", ...]
  {"End-Position": ["Place", "Person", "Entity"], ...}
  ```

### Model Training

Training scripts as follows:

- `run_seq2seq.py`: Python code entry, modified from the transformers/examples/seq2seq/run_seq2seq.py
- `run_seq2seq.bash`: Model training script logging to the log file.
- `run_seq2seq_verbose.bash`: Same model training script as `run_seq2seq.bash` but output to the screen directly.
- `run_seq2seq_with_pretrain.bash`: Model training script for curriculum learning, which contains substructure learning and full structure learning.

The command for the training is as follows (see bash scripts and Python files for the corresponding command-line
arguments):

```bash
bash run_seq2seq_verbose.bash -d 0 -f tree -m t5-base --label_smoothing 0 -l 1e-4 --lr_scheduler linear --warmup_steps 2000 -b 16
```

- `-d` refers to the GPU device id.
- `-m t5-base` refers to using T5-base.
- Currently, constrained decoding algorithms do not support `use_fast_tokenizer=True` and beam search yet.

Trained models are saved in the `models/` folder.

### Model Evaluation

Offset-level Evaluation

```bash
python evaluation.py -g <data-folder-path> -r <offset-folder-path> -p <model-folder-path> -f <data-format>
```
- This evaluation script converts the `eval_preds_seq2seq.txt` and `test_preds_seq2seq.txt` in the model folder `<model-folder-path>` into the corresponding offset prediction results for model evaluation.
- ``-f <data-format>`` refers to `dyiepp` or `oneie`

Record-level Evaluation (approximate, used in training)

```bash
bash run_eval.bash -d 0 -m <model-folder-path> -i <data-folder-path> -c -b 8
```

- `-d` refers to the GPU device id.
- `-c` represents the use of constrained decoding, otherwise not apply
- `-b 8` represents `batch_size=8`

### How to expand to other tasks

1. prepare the corresponding data format
2. Writ the code for reading corresponding data format: `elif data_args.task.startswith("event")` in `seq2seq.py`
3. Writ the code for evaluating the corresponding task result: `def compute_metrics(eval_preds)` in `seq2seq.py`

Completing the above process can finish the simple Seq2Seq training and inference process.

If you need to use constrained decoding, you need to write the corresponding decoding mode (decoding_format), refer to `extraction.extract_constraint.get_constraint_decoder`

## Pre-trained Model

You can find the pre-trained models as following google drive links or download models using command `gdown` (`pip install gdown`).

[dyiepp_ace2005_en_t5_base.zip](https://drive.google.com/file/d/1_fOmnSatNfceL9DZPxpof5AT9Oo7vTrC/view?usp=sharing)
```bash
gdown --id 1_fOmnSatNfceL9DZPxpof5AT9Oo7vTrC && unzip dyiepp_ace2005_en_t5_base.zip
```

[dyiepp_ace2005_en_t5_large.zip](https://drive.google.com/file/d/10iY1obkbgJtTKwfoOFevqL5AwG-hLvhU/view?usp=sharing)
```bash
gdown --id 10iY1obkbgJtTKwfoOFevqL5AwG-hLvhU && unzip dyiepp_ace2005_en_t5_large.zip
```

[oneie_ace2005_en_t5_large.zip](https://drive.google.com/file/d/1zwnptRbdZntPT4ucqSANeaJ3vvwKliUe/view?usp=sharing)
```bash
gdown --id 1zwnptRbdZntPT4ucqSANeaJ3vvwKliUe && unzip oneie_ace2005_en_t5_large.zip
```

[oneie_ere_en_t5_large.zip](https://drive.google.com/file/d/1WG7-pTZ3K49VMbQIONaDq_0pUXAcoXrZ/view?usp=sharing)
```bash
gdown --id 1WG7-pTZ3K49VMbQIONaDq_0pUXAcoXrZ && unzip oneie_ere_en_t5_large.zip
```

## Event Datasets Preprocessing

We first refer to the following code and environments [[dygiepp](https://github.com/dwadden/dygiepp)] and [[oneie v0.4.7](http://blender.cs.illinois.edu/software/oneie/)] for data preprocessing.
Thanks to them！

### DYGIEPP ACE05
Generated Path: `dygiepp/data/ace-event/processed-data/default-settings/json/`

Preprocessing Script:
``` bash
python ./scripts/data/ace-event/parse_ace_event.py default-settings
```

The version of spacy is `2.0.18`, it may affect results of sentence splitting. And the version of dygiepp I used as follows
```
commit edec203b73d32824f14e03b5510e020130b69a7f (HEAD -> master)
Author: dwadden <dwadden@cs.washington.edu>
Date: Sun Oct 11 15:02:33 2020 -0700

Add `dataset` argument when creating new doc-key.
```

### OneIE ACE05+
``` bash
ACE_DATA_FOLDER=<ACE_PATH>
mkdir -p data/ace05-EN
python preprocessing/process_ace.py -i ${ACE_DATA_FOLDER}/data -o data/ace05-EN -s resource/splits/ACE05-E -b bert-large-cased -l english
wc -l data/ace05-EN/*
```

- `nltk==3.5` is used in our experiments, we found `nltk==3.6+` may leads different sentence numbers.


After data preprocessing and we get the following data files:

```text
 $ tree data/raw_data/
data/raw_data/
├── ace05-EN
│   ├── dev.oneie.json
│   ├── test.oneie.json
│   └── train.oneie.json
├── dyiepp_ace2005
│   ├── dev.json
│   ├── test.json
│   └── train.json
└── ERE-EN
    ├── dev.oneie.json
    ├── test.oneie.json
    └── train.oneie.json
```

We then convert the above data files to tree format.
The following scripts generate the corresponding data folder in `data/text2tree`.
The conversion will automatically generate `train/dev/test` JSON files and `event.schema` file.

```bash
bash scripts/processing_data.bash
```

```text
data/text2tree
├── dyiepp_ace2005_subtype
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── dyiepp_ace2005_subtype_span
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── one_ie_ace2005_subtype
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── one_ie_ace2005_subtype_span
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── one_ie_ere_en_subtype
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
└── one_ie_ere_en_subtype_span
    ├── event.schema
    ├── test.json
    ├── train.json
    └── val.json
```

- `dyiepp_ace2005_subtype` for Full Structure Learning and `dyiepp_ace2005_subtype_span` for Substructure Learning.

## Citation

If this repository helps you, please cite this paper:

Yaojie Lu, Hongyu Lin, Jin Xu, Xianpei Han, Jialong Tang, Annan Li, Le Sun, Meng Liao, Shaoyi Chen. Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction. The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021).

```
@inproceedings{lu-etal-2021-text2event,
    title = "{T}ext2{E}vent: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction",
    author = "Lu, Yaojie  and
      Lin, Hongyu  and
      Xu, Jin  and
      Han, Xianpei  and
      Tang, Jialong  and
      Li, Annan  and
      Sun, Le  and
      Liao, Meng  and
      Chen, Shaoyi",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.217",
    pages = "2795--2806",
    abstract = "Event extraction is challenging due to the complex structure of event records and the semantic gap between text and event. Traditional methods usually extract event records by decomposing the complex structure prediction task into multiple subtasks. In this paper, we propose Text2Event, a sequence-to-structure generation paradigm that can directly extract events from the text in an end-to-end manner. Specifically, we design a sequence-to-structure network for unified event extraction, a constrained decoding algorithm for event knowledge injection during inference, and a curriculum learning algorithm for efficient model learning. Experimental results show that, by uniformly modeling all tasks in a single model and universally predicting different labels, our method can achieve competitive performance using only record-level annotations in both supervised learning and transfer learning settings.",
}
```
