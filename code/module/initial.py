import torch.nn as nn

def initial_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, (nn.LSTM)):
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)
            nn.init.zeros_(m.bias_hh_l0)
        elif isinstance(m, (nn.MultiheadAttention)):
            nn.init.xavier_normal_(m.out_proj.weight.data)
            m.out_proj.bias.data.zero_()