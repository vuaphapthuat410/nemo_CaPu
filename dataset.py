import traceback

from tqdm import tqdm
import torch
import logging


PUNCT_LABEL_TO_ID = {
    "O": 0,
    ",": 1,
    ".": 2,
    "?": 3
}
PUNC_ID_TO_LABEL = {
    0: "",
    1: ",",
    2: ".",
    3: "?"
}
CAP_LABEL_TO_ID = {
    "O": 0,
    "U": 1
}
CAP_ID_TO_LABEL = {
    0: "O",
    1: "U"

}

logging.basicConfig(level=logging.INFO)


class CapuDataset:
    def __init__(self, text_path=None, label_path=None, tokenizer=None, max_len=512, infer=False,
                 infer_text_ls=list, max_sample=None):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.logger = logging.getLogger(__name__)
        self.all_samples = []
        if not infer:
            assert text_path is not None and label_path is not None, 'Training mode require text path and label'
            with open(text_path) as f:
                text_data = f.read().splitlines()

            with open(label_path) as f:
                label_data = f.read().splitlines()
            if max_sample is not None:
                text_data = text_data[:max_sample]
                label_data = label_data[:max_sample]

            self.text_data = text_data
            self.label_data = label_data

            self.init_training_data()
        else:
            assert infer_text_ls, 'Infer mode require infer_text_ls != None and != empty list'
            self.infer_text_ls = infer_text_ls
            self.init_infer_data()


    def init_infer_data(self):
        self.logger.info('Init infer dataset')

        try:
            for text in tqdm(self.infer_text_ls):
                tokenize_res = self.tokenizer(text, padding='max_length', truncation=True, return_attention_mask=True,
                                              return_tensors='pt', max_length=self.max_len)

                input_ids = tokenize_res.input_ids.squeeze()
                attention_mask = tokenize_res.attention_mask.squeeze()

                self.all_samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

            self.logger.info(f"Len infer sample:  {len(self.all_samples)}")

        except Exception:
            self.logger.error('Some example fail')

    def init_training_data(self):
        self.logger.info('Init training dataset')

        fail_sample = 0
        for text, label in tqdm(zip(self.text_data, self.label_data)):
            try:
                label_ls = label.split()
                punct_labels = [PUNCT_LABEL_TO_ID[i[0]] for i in label_ls]
                capital_labels = [CAP_LABEL_TO_ID[i[1]] for i in label_ls]
                assert len(punct_labels) == len(capital_labels), 'len label mismatch'
                assert len(punct_labels) == len(text.split()), 'len label text mis match'
                punct_labels_ex = []
                capital_labels_ex = []
                attention_mask = []
                input_ids = []
                i = -1
                count = 0
                tokens_ls = self.tokenizer.tokenize(text)[:self.max_len - 2]
                for ind, token_i in enumerate(tokens_ls):
                    if token_i.startswith('▁'):
                        i += 1
                        count += 1

                    id_i = self.tokenizer.convert_tokens_to_ids(token_i)

                    input_ids.append(id_i)
                    attention_mask.append(1)
                    punct_labels_ex.append(punct_labels[i])
                    capital_labels_ex.append(capital_labels[i])

                punct_labels_ex.append(0)
                capital_labels_ex.append(0)
                attention_mask.append(1)
                input_ids.append(2)

                punct_labels_ex.insert(0,0)
                capital_labels_ex.insert(0,0)
                attention_mask.insert(0,1)
                input_ids.insert(0,0)

                # assert count == len(punct_labels_ex) - 2, 'len mis match _token with len label'

                while len(input_ids) < self.max_len:
                    punct_labels_ex.append(0)
                    capital_labels_ex.append(0)
                    attention_mask.append(0)
                    input_ids.append(1)

                # tta = ''
                # for tk, p_lb, c_lb in zip(tokens_ls, punct_labels_ex[1:], capital_labels_ex[1:]):
                #     tt = tk + PUNC_ID_TO_LABEL[p_lb] + ' '
                #     if c_lb == 'U':
                #         tt = tt.title()
                #     tta += tt
                # print(tta)
                # input()
                punct_labels_ex = torch.Tensor(punct_labels_ex)
                capital_labels_ex = torch.Tensor(capital_labels_ex)
                attention_mask = torch.Tensor(punct_labels_ex)
                input_ids = torch.Tensor(input_ids)


                self.all_samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "capital_labels": capital_labels_ex,
                    "punctuation_labels": punct_labels_ex
                })
            except Exception as ex:
                fail_sample += 1

        print("[INFO] len training sample: ", len(self.all_samples))
        self.logger.info(f'Len sample fail:{fail_sample}')

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, i):
        return self.all_samples[i]
