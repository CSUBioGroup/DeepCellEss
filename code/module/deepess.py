import torch.nn as nn
import torch.nn.functional as F


class DeepEss(nn.Module):
    def __init__(self, max_len, fea_size, kernel_size, num_head, hidden_size, num_layers, attn_drop, lstm_drop, linear_drop,
                 structure='TextCNN+MultiheadAttn+BiLSTM+Maxpool+MLP', name='DeepEss'):
        super(DeepEss, self).__init__()
        self.structure = structure
        self.name = name
        self.textCNN = nn.Conv1d(in_channels=fea_size,
                                 out_channels=fea_size,
                                 kernel_size=kernel_size,
                                 padding='same')
        self.multiAttn = nn.MultiheadAttention(embed_dim=fea_size,
                                               num_heads=num_head,
                                               dropout=attn_drop,
                                               batch_first=True)
        self.layerNorm = nn.LayerNorm(fea_size)
        self.biLSTM = nn.LSTM(fea_size,
                              hidden_size,
                              bidirectional=True,
                              batch_first=True,
                              num_layers=num_layers,
                              dropout=lstm_drop)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.generator = nn.Sequential(nn.Linear(hidden_size * 2, 1),
                                       nn.Dropout(linear_drop),
                                       nn.Sigmoid())
        
    def forward(self, x, get_attn=False):
        # => batch_size × seq_len × fea_size
        residual = x
        x = x.permute(0, 2, 1)
        # => batch_size × fea_size × seq_len
        x = F.relu(self.textCNN(x))
        x = residual + x.permute(0, 2, 1)
        # => batch_size × seq_len × fea_size, batch_size × seq_len × seq_len
        attn_output, seq_attn = self.multiAttn(x, x, x)
        x = x + self.layerNorm(attn_output)
        # => batch_size × seq_len × hidden_size*2
        x, _ = self.biLSTM(x)
        x = x.permute(0, 2, 1)
        x = self.generator(self.pool(x).squeeze(-1))
        if get_attn == True:
            # => batch_size × 1, batch_size × seq_len × seq_len
            return x, seq_attn
        else:
            # => batch_size × 1
            return x