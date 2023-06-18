import torch
import torch.nn as nn
import random

class EncoderNet(nn.Module):
    def __init__(self, fp_size, encoding_size):
        super(EncoderNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(fp_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, encoding_size)
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        #out.shape = [batch_size, 256]
        return out

class DecoderNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(DecoderNet, self).__init__()

        # GRU parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # output token count
        self.output_size = output_size

        # pytorch.nn
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        #x.shape = [batch_size, selfie_len, encoding_size] = [64, 128, 256]
        out, h = self.gru(x, h)
        #out.shape = [batch_size, selfie_len, hidden_size] = [64, 128, 256]
        #h.shape = [num_layers, batch_size, hidden_size] = [1, 64, 256]
        return out, h

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0
    
class EncoderDecoder(nn.Module):
    def __init__(self, fp_size=4860, encoding_size=256, hidden_size=256, num_layers=2, output_size=42, dropout=0.2,
                teacher_forcing=True, teacher_ratio=0.3, random_seed=42):
        super(EncoderDecoder, self).__init__()
        self.encoding_size = encoding_size
        self.hidden_size = encoding_size + 42
        self.teacher_forcing = teacher_forcing
        self.teacher_ratio = teacher_ratio
        self.encoder = EncoderNet(fp_size, encoding_size)
        self.decoder = DecoderNet(self.hidden_size, self.hidden_size, num_layers, output_size, dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        random.seed(random_seed)

        #pytorch.nn
        self.fc = nn.Linear(self.hidden_size, 42)
        self.relu = nn.ReLU()
        self.softmax2d = nn.Softmax(dim=2)

    def forward(self, x, y):
        hidden = self.decoder.init_hidden(batch_size=x.shape[0]).to(self.device)
        encoded = self.encoder(x) # shape [batch_size, encoding_size]
        
        start_token = self.generate_start_token(batch_size=x.shape[0]) # shape [batch_size, 42]
        gru_input0 = torch.cat([encoded, start_token], dim=1)
        
        x = gru_input0.unsqueeze(1)
        decoded = []
        for n in range(128):
            out, hidden = self.decoder(x, hidden)
            out = self.relu(self.fc(out)) # [batch_size, 1, hidden_size] -> [batch_size, 1, 42]
            decoded.append(out)
            random_float = random.random()
            if self.teacher_forcing and random_float < self.teacher_ratio:
                out = y.unsqueeze(1)[:,:,n]
            x = torch.cat([encoded.unsqueeze(1), out], dim=2)
        out_cat = torch.cat(decoded, dim=1)
        out_cat = self.softmax2d(out_cat)
        return out_cat # shape [batch_size, selfie_len, alphabet_len]
    
    def generate_start_token(self, batch_size, idx=41):
        start = torch.zeros(42).to(self.device)
        start[idx] = 1
        start = start.unsqueeze(0).repeat(batch_size, 1)
        return start