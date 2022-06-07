# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The script converts raw text to the NeMo format for punctuation and capitalization task.
Raw Data Format
---------------
The Punctuation and Capitalization model can work with any text dataset, although it is recommended to balance the data, especially for the punctuation task.
Before pre-processing the data to the format expected by the model, the data should be split into train.txt and dev.txt (and optionally test.txt).
Each line in the **train.txt/dev.txt/test.txt** should represent one or more full and/or truncated sentences.
Example of the train.txt/dev.txt file:
    When is the next flight to New York?
    The next flight is ...
    ....
The `source_data_dir` structure should look like this:
   .
   |--sourced_data_dir
     |-- dev.txt
     |-- train.txt
NeMo Data Format for training the model
---------------------------------------
The punctuation and capitalization model expects the data in the following format:
The training and evaluation data is divided into 2 files: text.txt and labels.txt. \
Each line of the **text.txt** file contains text sequences, where words are separated with spaces, i.e.
[WORD] [SPACE] [WORD] [SPACE] [WORD], for example:
        when is the next flight to new york
        the next flight is ...
        ...
The **labels.txt** file contains corresponding labels for each word in text.txt, the labels are separated with spaces. \
Each label in labels.txt file consists of 2 symbols:
* the first symbol of the label indicates what punctuation mark should follow the word (where O means no punctuation needed);
* the second symbol determines if a word needs to be capitalized or not (where U indicates that the word should be upper cased, and O - no capitalization needed.)
By default the following punctuation marks are considered: commas, periods, and question marks; the rest punctuation marks were removed from the data.
This can be changed by introducing new labels in the labels.txt files
Each line of the labels.txt should follow the format: [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt). \
For example, labels for the above text.txt file should be:
        OU OO OO OO OO OO OU ?U
        OU OO OO OO ...
        ...
The complete list of all possible labels for this task used in this tutorial is: OO, ,O, .O, ?O, OU, ,U, .U, ?U.
Converting Raw data to NeMo format
----------------------------------
To pre-process the raw text data, stored under :code:`sourced_data_dir` (see the :ref:`raw_data_format_punct`
section), run the following command:
    python examples/nlp/token_classification/data/prepare_data_for_punctuation_capitalization.py \
           -s <PATH/TO/THE/SOURCE/FILE> \
           -o <PATH/TO/THE/OUTPUT/DIRECTORY>
"""

import argparse
import os
import string
import re




def remove_punctuation(word: str):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    # all_punct_marks = string.punctuation.replace("'", '')
    all_punct_marks = '!,.?:;'
    return re.sub('[' + all_punct_marks + ']', '', word)



def create_text_and_labels(output_dir: str, file_path: str, punct_marks: str = ',.?'):
    """
    Create datasets for training and evaluation.

    Args:
      output_dir: path to the output data directory
      file_path: path to file name
      punct_marks: supported punctuation marks

    The data will be split into 2 files: text.txt and labels.txt. \
    Each line of the text.txt file contains text sequences, where words\
    are separated with spaces. The labels.txt file contains \
    corresponding labels for each word in text.txt, the labels are \
    separated with spaces. Each line of the files should follow the \
    format:  \
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
    """
    if not os.path.exists(file_path):
        raise ValueError(f'{file_path} not found')

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    labels_file = os.path.join(output_dir, 'labels_' + base_name)
    text_file = os.path.join(output_dir, 'text_' + base_name)

    with open(file_path, 'r') as f:
        with open(text_file, 'w') as text_f:
            with open(labels_file, 'w') as labels_f:
                for line in f:
                    if not line:
                        continue
                    line = line.split()
                    text = ''
                    labels = ''
                    for word in line:
                        label = word[-1] if word[-1] in punct_marks else 'O'
                        if  word[-1] == ';':
                            label = ','
                        if  word[-1] == '!':
                            label = '.'
                        word = remove_punctuation(word)
                        if len(word) > 0:
                            if word[0].isupper():
                                label += 'U'
                            else:
                                label += 'O'

                            word = word.lower()
                            text += word + ' '
                            labels += label + ' '

                    text_f.write(text.strip() + '\n')
                    labels_f.write(labels.strip() + '\n')

    print(f'{text_file} and {labels_file} created from {file_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for punctuation and capitalization tasks')
    parser.add_argument("-s", "--source_file", required=True, type=str, help="Path to the source file")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()

    if not os.path.exists(args.source_file):
        raise ValueError(f'{args.source_file} was not found')

    os.makedirs(args.output_dir, exist_ok=True)
    create_text_and_labels(args.output_dir, args.source_file)

    print(f'Processing of the {args.source_file} is complete')
