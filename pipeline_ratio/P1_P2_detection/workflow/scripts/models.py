import torch
from torch import nn
from itertools import starmap
import yaml
from yaml.loader import SafeLoader
import torchvision

dims = '../config/models_dims.yaml'
params = '../config/params.yaml'

with open(dims) as f:
    config = yaml.load(f, Loader=SafeLoader)

    # batch norm and dropout parameters
    EPS = config['eps']
    MOM = config['mom']
    DROP = config['drop']

    # CNN parameters
    PAD = config['padding']
    STRD = config['stride']
    CHS = config['channels']
    KER = config['kernels']

    # LSTM parameters
    HDS = config['hidden_layer_size']
    NUML = config['num_cells']

    # MLP parameters
    HID = config['hidden_layers']
    WID = config['width']

with open(params) as f:
    config = yaml.load(f, Loader=SafeLoader)

    # input size
    SIZE = config['size']


### CNN model

class Block(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch,
                 ker):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,
                               out_ch,
                               kernel_size=ker,
                               stride=STRD,
                               padding=PAD)
        
        self.batch_norm = nn.BatchNorm1d(num_features=out_ch,
                                         eps=EPS, 
                                         momentum=MOM)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        return self.relu(x)
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = starmap(Block, zip(CHS, CHS[1:], KER))
        self.enc_blocks = nn.ModuleList(self.l)
        self.pool       = nn.AdaptiveMaxPool1d(1)
        self.relu  = nn.ReLU()
            
    def forward(self, x):
            for block in self.enc_blocks:
                x = block(x)
                x = self.pool(x)
            return x
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CHS[-1], SIZE),
            nn.ReLU(),
            nn.Dropout(DROP),
            nn.Linear(SIZE, SIZE),
        )
    
    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        x = self.linear_relu_stack(x)
        return x.squeeze().sigmoid()

#### MLP baseline

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layer = (nn.Linear(WID, WID), nn.ReLU())
        self.layers = layer * HID
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(SIZE, WID),
            nn.BatchNorm1d(num_features=WID,
                                         eps=EPS, 
                                         momentum=MOM),
            nn.ReLU(),
            *self.layers,
            nn.Dropout(DROP),
            nn.Linear(WID, SIZE),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x.sigmoid()
   
### Bidirectional LSTM
 
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_layer_size = HDS
        self.lstm = nn.LSTM(SIZE, 
                            hidden_size=HDS,
                            num_layers=NUML,
                            batch_first=True, 
                            bidirectional=True)
        self.hidden = nn.Linear(2*HDS, 2*HDS)
        self.linear = nn.Linear(2*HDS, SIZE)
        self.dropout = nn.Dropout(DROP)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x)
        x = self.hidden(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.linear(lstm_out.squeeze())
        return y
    
### LSTM-FCN

class BlockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=SIZE, 
                            hidden_size=HDS, 
                            bidirectional=True)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x, self.hidden_cell = self.lstm(x)
        return x

class LSTMFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_block = BlockLSTM()
        self.fcn_block = Encoder()
        self.linear = nn.Linear(2*HDS + CHS[-1], SIZE)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.lstm_block(x).squeeze()
        x2 = self.fcn_block(x).squeeze()
        x = torch.cat([x1, x2], 1)
        y = self.linear(x)
        return y
    
#### LSTM AE

class LSTM_Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=128):
    super(LSTM_Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.unsqueeze(-1)
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.squeeze()
  
class LSTM_Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=128, n_features=1):
    super(LSTM_Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.lin_layer = nn.Linear(self.hidden_dim, 1)
    self.output_layer = nn.Linear(self.seq_len, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.unsqueeze(1)
    x = x.repeat(1, self.seq_len, self.n_features)
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = self.lin_layer(x).squeeze()
    return x
  
class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len=SIZE, n_features=1, embedding_dim=128):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = LSTM_Encoder(seq_len, n_features, embedding_dim)
    self.decoder = LSTM_Decoder(seq_len, embedding_dim, n_features)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  
#UNet
class UBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=2)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3)
        self.batch_norm = nn.BatchNorm1d(num_features=out_ch,
                                    eps=EPS, 
                                    momentum=MOM)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.batch_norm(out1)
        out_relu = self.relu(out1)
        out_relu = self.dropout(out_relu)
        out2 = self.conv2(out_relu)
        return out2

class UEncoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.l = starmap(UBlock, zip(chs, chs[1:]))
        self.enc_blocks = nn.ModuleList(self.l)
        self.pool       = nn.MaxPool1d(2)
        
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class UDecoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs        = chs
        self.conv       = lambda x, y: nn.ConvTranspose1d(x, y, 2, 2)
        self.upconvs    = nn.ModuleList(starmap(self.conv, zip(chs, chs[1:])))
        self.dec_blocks = nn.ModuleList(starmap(UBlock, zip(chs, chs[1:])))
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
            x = self.dropout(x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(self, 
                 enc_chs=(1, 16, 32, 64), 
                 dec_chs=(64, 32, 16), 
                 num_class=1):
        
        super().__init__()
        self.encoder     = UEncoder(enc_chs)
        self.decoder     = UDecoder(dec_chs)
        self.head        = nn.Conv1d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[-1], enc_ftrs[::-1])
        out      = self.head(out)
        return out.squeeze()