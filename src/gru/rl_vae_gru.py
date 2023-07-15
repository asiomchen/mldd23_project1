import torch
import torch.nn as nn
import random
from src.utils.vectorizer import SELFIESVectorizer
import torch.nn.functional as F
import rdkit.Chem as Chem
import rdkit.Chem.QED
import selfies as sf

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

    def init_hidden(self, batch_size, batched=True):
        if batched:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, self.hidden_size)
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

    def forward(self, x, y, teacher_forcing, reinforcement=False):
        batch_size = x.shape[0]
        hidden = self.decoder.init_hidden(batch_size).to(self.device)
        mu, logvar = self.encoder(x)
        encoded = self.reparameterize(mu, logvar)
        x = encoded.unsqueeze(1)
        outputs = []
        
        # generating sequence
        
        for n in range(128):
            out, hidden = self.decoder(x, hidden)
            out = self.relu(self.fc1(out)) # shape (batch_size, 42)
            outputs.append(out)
            random_float = random.random()
            if teacher_forcing and random_float < self.teacher_ratio:
                out = y[:,n,:].unsqueeze(1)
            x = self.relu(self.fc2(out))
        out_cat = torch.cat(outputs, dim=1)
        #out_cat = self.softmax2d(out_cat)
        
        if reinforcement:
            rl_loss, total_reward = self.reinforce(out_cat)
            return out_cat, rl_loss, total_reward
        else:
            return out_cat # out_cat.shape [batch_size, selfie_len, alphabet_len]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def reinforce(self, out_cat, n_samples=10):
        
        vectorizer = SELFIESVectorizer(pad_to_len=128)
        out_cat = out_cat.detach().cpu().numpy()
        
        # reinforcement learning    
        rl_loss = 0
        total_reward = 0
        gamma = 0.97
        batch_size = out_cat.shape[0]
        total_reward = 0
        
        succesful = False
        # check if able to decode all, else reroll
        while succesful == False:
            sample_idxs = [random.randint(0, batch_size-1) for x in range(n_samples)]
            for idx in sample_idxs:
                try:
                    seq = vectorizer.devectorize(out_cat[idx], remove_special=True)
                    _ = sf.decoder(seq)
                    succesful = True
                except:
                    pass
        
        for idx in sample_idxs:
            # get reward
            trajectory = vectorizer.devectorize(out_cat[idx], remove_special=True)
            trajectory = sf.decoder(trajectory)
            reward = self.get_reward(trajectory)
            
            # convert string of characters into tensor of shape [selfie_len, alphabet_len]
            trajectory_input = vectorizer.vectorize(trajectory)
            trajectory_input = torch.tensor(trajectory_input)
            discounted_reward = reward
            total_reward += reward
            
            # init hidden layer
            hidden = self.decoder.init_hidden(batch_size, batched=False).to(self.device)
            
            # 'follow' the trajectory
            for p in range(len(trajectory)-1):
                token = trajectory_input[p].detach().cpu()
                argmax = torch.argmax(token).item()
                print(argmax)
                token = token.float().to(self.device)
                while (argmax != 40 and argmax != 39): # until encounters [nop] or [end]
                    representation = self.relu(self.fc2(token)).unsqueeze(0)
                    representation, hidden = self.decoder(representation, hidden)
                    representation = representation.squeeze(0)
                    next_token = self.relu(self.fc1(representation))
                    log_probs = F.log_softmax(next_token, dim=1)
                    top_i = trajectory_input[p+1]
                    rl_loss -= (log_probs[0, top_i]*discounted_reward)
                    discounted_reward = discounted_reward * gamma
        rl_loss = rl_loss / n_samples
        total_reward = total_reward / n_samples
        return rl_loss, total_reward
    
    def get_reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        reward = Chem.QED.qed(mol)
        return reward