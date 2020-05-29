import sys
import torch
sys.path.append('/Users/gabrielepicco/Documents/WH-GovHHS/GovHHS-Analytics-cca-neural-reasoning')
from spectre.module.backward_chaining_inference import BackwardChainingInference
from transformers import BertTokenizer, BertConfig
from bertviz.neuron_view import show
from bertviz.transformers_neuron_view.modeling_bert import BertModel

import os
from os.path import expanduser
home = expanduser("~")
TEACHER_PATH = os.path.join(home,"Downloads/reasoner_00.model")
UNIFIER_PATH = os.path.join(home,"Downloads/unification_01.model")

device = 'cpu'
model = BackwardChainingInference().to(device)
model.load_state_dict(torch.load(UNIFIER_PATH, map_location=device))
model.teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))

model_type = 'bert'
model_version = "bert-base-uncased"
config = BertConfig.from_pretrained(model_version, output_attentions=True)
unification_custom = BertModel.from_pretrained(model_version, config=config)
unification_custom.load_state_dict(model.unification.model.state_dict())
model.unification.model = unification_custom

context = "The dog is blue. If someone is nice and they chase the lion then they visit the bear. If someone chases the dog then they visit the bear. If the bear visits the lion then the lion needs the dog. If someone needs the lion and the lion visits the bear then the lion chases the dog. If someone visits the bear then the bear visits the dog. The bear chases the dog. If someone chases the lion and they visit the bear then the lion visits the dog. The lion visits the dog."
question = "The bear visits the dog." # True
context_question_embeds, context_embeds, masks, types = model.unification.data_prep([context], [question])
attn_data = model.unification.model(inputs_embeds=context_question_embeds.to(model.unification.device), attention_mask=masks.to(model.unification.device), token_type_ids=types.to(model.unification.device))[-1]

show(attn_data, model_type=model_type, tokenizer=model.unification.tokenizer, sentence_a=context, sentence_b=question)
