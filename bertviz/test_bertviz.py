from transformers import BertTokenizer, BertConfig

from bertviz.neuron_view import show
from bertviz.transformers_neuron_view.modeling_bert import BertModel

model_type = 'bert'
model_version = 'bert-base-uncased'
do_lower_case = True

config = BertConfig.from_pretrained(model_version, output_attentions=True)
model = BertModel.from_pretrained(model_version, config=config)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
#call_html()
show(model, model_type, tokenizer, sentence_a, sentence_b)