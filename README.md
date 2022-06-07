
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     | '.
                 / \\|||  :  ||| \
                / _||||| -:- |||||- \
               |   | \\\  -  / |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project is blessed by Buddha. Never generates bugs

# Collect and build supervised data (remove punctuation and lower text to create data)
(Data is fetched from original set, located at 172.16.10.201:/home/linhnguyen/news)
- mkdir data
- python handle_dataset.py

# Prepare data to make label file (follow nemo format data label):
One txt label file contain multiple lines, each of which is label of one training sample (a sequence)
In each row, a single label consists of 2 symbols:

   + the first symbol of the label indicates what punctuation mark should follow the word (where O means no punctuation needed);
   + the second symbol determines if a word needs to be capitalized or not (where U indicates that the word should be upper cased, and O - no capitalization needed.)
  In this tutorial, we are considering only commas, periods, and question marks the rest punctuation marks were removed. To use more punctuation marks, update the dataset to include desired labels, no changes to the model needed.

  - Each line of the labels.txt should follow the format: [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt). For example, labels for the above text.txt file should be:

  OU OO OO OO OO OO OU ?U
  OU OO OO OO ...
   ...
  - The complete list of all possible labels for this task used in this tutorial is: OO, ,O, .O, ?O, OU, ,U, .U, ?U.

RUN: python prepare_data_for_punctuation_capitalization.py --s data/train.txt --o data/preprocessed
# Training:
- mkdir checkpoints
- python train_capu.py  --dataset_tag <foldername_to_save_checkpoint>
# Infer:
- python model.py