from transformers import BertModel, BertPreTrainedModel
import torch


class FCLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=None):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        print("activation: ", self.use_activation)
        
    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.use_activation(x)
        return self.linear(x)


class BertSts(BertPreTrainedModel):
    def __init__(self, config, add_fc = False) -> None:
        super(BertSts, self).__init__(config)
        self.bert = BertModel(config)
        self.output_layer = FCLayer(config.hidden_size, 1)
        self.dense = torch.nn.Sequential(*add_fc) if add_fc else False

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.dense:
            dense_outputs = self.dense(bert_outputs['pooler_output'])
            sim_score = self.output_layer(dense_outputs)
        else:
            sim_score = self.output_layer(bert_outputs['pooler_output'])
        return sim_score


def init_fclayer(in_size, out_size, layer, neuron, dropout):
    assert layer == len(neuron)+1, "추가할 레이어의 수가 레이어별 뉴런 수보다 1 많아야 합니다."
    st = [in_size]
    ed = [out_size]
    if layer == 1:
        dense = [FCLayer(in_size, out_size, dropout, torch.nn.GELU())]
    else:
        neurons = st + neuron + ed
        dense = [FCLayer(neurons[i], neurons[i+1], dropout, torch.nn.GELU()) for i in range(layer)]
    return dense