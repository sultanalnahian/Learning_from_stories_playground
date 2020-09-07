from __future__ import absolute_import, division, print_function

import glob
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import pearsonr

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (convert_examples_to_features,
                        output_modes, processors, InputExample)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

class ggModel():

    def __init__(self):
        self.args = {
            'data_dir': 'data/',
            'model_type':  'bert',
            'model_name': 'bert-base-cased',
            'task_name': 'binary',
            'output_dir': 'outputs/',
            'cache_dir': 'cache/',
            'do_train': True,
            'do_eval': True,
            'fp16': True,
            'fp16_opt_level': 'O1',
            'max_seq_length': 128,
            'output_mode': 'classification',
            'train_batch_size': 8,
            'eval_batch_size': 1,

            'gradient_accumulation_steps': 1,
            'num_train_epochs': 4,
            'weight_decay': 0,
            'learning_rate': 4e-5,
            'adam_epsilon': 1e-8,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,

            'logging_steps': 50,
            'evaluate_during_training': False,
            'save_steps': 2000,
            'eval_all_checkpoints': True,

            'overwrite_output_dir': True,
            'reprocess_input_data': True,
            'notes': 'Using Yelp Reviews dataset'
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args['model_type']]
        
        self.config = config_class.from_pretrained(self.args['model_name'], num_labels=2, finetuning_task=self.args['task_name'])
        self.tokenizer = tokenizer_class.from_pretrained(self.args['model_name'])
        checkpoints = [self.args['output_dir']]
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(self.args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        checkpoint = checkpoints[0]
        self.model = model_class.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""


    def load_and_cache_examples(self, task, tokenizer, sentence, evaluate=False):
        task = self.args['task_name']
        processor = processors[task]()
        output_mode = self.args['output_mode']
        label = '1'
        set_type = "dev"
        guid = "%s-%s" % (set_type, 0)
        text_a = sentence
        examples = []
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        label_list = processor.get_labels()
        features = convert_examples_to_features(examples, label_list, self.args['max_seq_length'], tokenizer, output_mode,
                cls_token_at_end=bool(self.args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                cls_token_segment_id=2 if self.args['model_type'] in ['xlnet'] else 0,
                pad_on_left=bool(self.args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
                pad_token_segment_id=4 if self.args['model_type'] in ['xlnet'] else 0)
            
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset


    def evaluate(self, sentence, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        
        EVAL_TASK = self.args['task_name']

        eval_dataset = self.load_and_cache_examples(EVAL_TASK, self.tokenizer, sentence, evaluate=True)
    
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        
        preds = None
        out_label_ids = None
#         for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        return preds

    def get_score(self, sentence):
        result = self.evaluate(sentence, prefix=self.global_step)
        return result