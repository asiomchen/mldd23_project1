import torch
import torch.nn as nn
import random

class VAEEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc41 = nn.Linear(512, output_size)
        self.fc42 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc41(h3)
        return mu, logvar

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
    def __init__(self, fp_size=4860, encoding_size=512, hidden_size=512, num_layers=0, output_size=42, dropout=0,
                teacher_ratio=0.5, random_seed=42):
        super(EncoderDecoder, self).__init__()
        self.teacher_ratio = teacher_ratio
        self.encoder = VAEEncoder(fp_size, encoding_size)
        self.decoder = DecoderNet(encoding_size, hidden_size, num_layers, output_size, dropout)
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        random.seed(random_seed)
        
        #pytorch.nn
        self.fc1 = nn.Linear(hidden_size, 42)
        self.fc2 = nn.Linear(42, hidden_size)
        self.relu = nn.ReLU()
        self.softmax2d = nn.Softmax(dim=2)

    def forward(self, x, y, teacher_forcing):
        hidden = self.decoder.init_hidden(batch_size=x.shape[0]).to(self.device)
        mu, logvar = self.encoder(x)
        encoded = self.reparametrize(mu, logvar)
        x = encoded.unsqueeze(1)
        decoded = []
        for n in range(128):
            out, hidden = self.decoder(x, hidden)
            out = self.relu(self.fc1(out))
            decoded.append(out)
            random_float = random.random()
            if teacher_forcing and random_float < self.teacher_ratio:
                out = y[:,n,:].unsqueeze(1)
            x = self.relu(self.fc2(out))
        out_cat = torch.cat(decoded, dim=1)
        out_cat = self.softmax2d(out_cat)
        return out_cat # shape [batch_size, selfie_len, alphabet_len]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)