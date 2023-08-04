import torch
import torch.nn as nn
import random
from src.utils.vectorizer import SELFIESVectorizer
import torch.nn.functional as F
import rdkit.Chem as Chem
import rdkit.Chem.QED
import selfies as sf
import pandas as pd


class VAEEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Encoder net, part of VAE.

        Parameters:
            input_size (int):size of the fingerprint vector
            output_size (int):size of the latent vectors mu and logvar
        """
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc41 = nn.Linear(512, output_size)
        self.fc42 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.tensor):fingerprint vector
        Returns:
            mu (torch.tensor): mean
            logvar: (torch.tensor): log variance
        """
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar


class DecoderNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(DecoderNet, self).__init__()
        """
        Decoder class based on GRU.
        
        Parameters:
            input_size (int):VAE encoder latent vector size
            hidden_size (int):GRU hidden size
            num_layers (int):GRU number of layers
            output_size (int):GRU output size (alphabet size)
            dropout (float):GRU dropout
        """
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

    def forward(self, x, h):
        """
        Args:
            x (torch.tensor):latent vector of size [batch_size, 1, input_size]
            h (torch.tensor):GRU hidden state of size [num_layers, batch_size, hidden_size]

        Returns:
            out (torch.tensor):GRU output of size [batch_size, 1, output_size]
        """
        out, h = self.gru(x, h)
        return out, h

    def init_hidden(self, batch_size, batched=True):
        if batched:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, self.hidden_size)
        return h0


class EncoderDecoder(nn.Module):
    def __init__(self, fp_size=4860, encoding_size=512, hidden_size=512, num_layers=1, output_size=42, dropout=0,
                 teacher_ratio=0.5, random_seed=42, use_cuda=True, nograd_encoder=False):
        """
        Encoder-Decoder class based on VAE and GRU.

        Parameters:
            fp_size (int): size of the fingerprint vector
            encoding_size (int): size of the latent vectors mu and logvar
            hidden_size (int): GRU hidden size
            num_layers (int): GRU number of layers
            output_size (int): GRU output size (alphabet size)
            dropout (float): GRU dropout
            teacher_ratio (float): teacher forcing ratio
            random_seed (int): random seed for reproducibility
            use_cuda (bool): wetter to use cuda
            nograd_encoder (bool): disable gradient calculation for the encoder
        """
        super(EncoderDecoder, self).__init__()
        self.teacher_ratio = teacher_ratio
        self.encoder = VAEEncoder(fp_size, encoding_size)
        self.decoder = DecoderNet(encoding_size, hidden_size, num_layers, output_size, dropout)
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.nograd_encoder = nograd_encoder
        random.seed(random_seed)

        # pytorch.nn
        self.fc1 = nn.Linear(hidden_size, 42)
        self.fc2 = nn.Linear(42, hidden_size)
        self.relu = nn.ReLU()
        self.softmax2d = nn.Softmax(dim=2)

    def forward(self, X, y, teacher_forcing=False, reinforcement=False):
        """
        Args:
            X (torch.tensor):batched fingerprint vector of size [batch_size, fp_size]
            y (torch.tensor):batched SMILES of target molecules
            teacher_forcing: (bool):enable teacher forcing
            reinforcement: (bool):enable loss calculation for use in reinforcement learning

        Returns:
            out_cat (torch.tensor):batched prediction tensor [batch_size, seq_len, alphabet_size]

        If reinforcement is enabled, the following tensors are also returned:
            rl_loss (torch.tensor):loss for use in reinforcement learning
            total_reward (torch.tensor):total reward for use in reinforcement learning
        """
        batch_size = X.shape[0]
        hidden = self.decoder.init_hidden(batch_size).to(self.device)

        if self.nograd_encoder:
            with torch.no_grad():
                mu, logvar = self.encoder(X)
                encoded = self.reparameterize(mu, logvar)
                x = encoded.unsqueeze(1)
        else:
            mu, logvar = self.encoder(X)
            encoded = self.reparameterize(mu, logvar)
            x = encoded.unsqueeze(1)
        outputs = []

        # generating sequence

        for n in range(128):
            out, hidden = self.decoder(x, hidden)
            out = self.relu(self.fc1(out))  # shape (batch_size, 42)
            outputs.append(out)
            random_float = random.random()
            if teacher_forcing and random_float < self.teacher_ratio:
                out = y[:, n, :].unsqueeze(1)
            x = self.fc2(out)
        out_cat = torch.cat(outputs, dim=1)

        if reinforcement:
            rl_loss, total_reward = self.reinforce(out_cat, X)
            return out_cat, rl_loss, total_reward
        else:
            return out_cat  # out_cat.shape [batch_size, selfie_len, alphabet_len]

    def reinforce(self, out_cat, X, n_samples=10):
        """
        Reinforcement learning loop for use in training the model.
        Args:
            out_cat (torch.tensor): batched vectorized SELFIES tensor
            X: (torch.tensor): batched fingerprint tensor of shape [batch_size, fp_size]
            n_samples (int): number of samples to draw from batch
        Returns:
            rl_loss (torch.tensor): loss for use in reinforcement learning
            total_reward (torch.tensor): total reward for use in reinforcement learning
        """
        vectorizer = SELFIESVectorizer(pad_to_len=128)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_cat = out_cat.detach().cpu().numpy()

        # reinforcement learning
        rl_loss = 0
        gamma = 0.97
        batch_size = out_cat.shape[0]
        total_reward = 0

        sample_idxs = [random.randint(0, batch_size - 1) for x in range(n_samples)]

        for idx in sample_idxs:
            # get reward
            trajectory = vectorizer.devectorize(out_cat[idx], remove_special=True)
            try:
                smiles = sf.decoder(trajectory)
                mol = Chem.MolFromSmiles(smiles)
                reward = (self.get_qed_reward(mol) + self.get_fp_reward(mol, X[idx])) / 2
            except sf.DecoderError:
                print('SELFIES decoding error')
                reward = 0

            # convert string of characters into tensor of shape [selfie_len, alphabet_len]
            trajectory_input = vectorizer.vectorize(trajectory)
            trajectory_input = torch.tensor(trajectory_input)
            discounted_reward = reward
            total_reward += reward

            # init hidden layer
            hidden = self.decoder.init_hidden(batch_size, batched=False).to(self.device)

            # 'follow' the trajectory
            for p in range(len(trajectory_input) - 1):
                token = trajectory_input[p]
                token_idx = torch.argmax(token.detach().cpu()).item()
                if token_idx in (39, 40):  # finish loss calculation if encounters [nop] or [end]
                    break
                token = token.float().to(device)
                representation = self.relu(self.fc2(token)).unsqueeze(0)  # [1, 512]
                representation, hidden = self.decoder(representation, hidden)
                representation = representation.squeeze(0)  # [512]
                next_token = self.relu(self.fc1(representation))  # [42]
                log_probs = F.log_softmax(next_token, dim=0)  # [42]
                top_i = torch.argmax(trajectory_input[p + 1])
                rl_loss -= (log_probs[top_i] * discounted_reward)
                discounted_reward = discounted_reward * gamma

        rl_loss = rl_loss / n_samples
        total_reward = total_reward / n_samples
        return rl_loss, total_reward

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Reparametrization trick for sampling from VAE latent space.
        Args:
            mu (torch.tensor): mean
            logvar: (torch.tensor): log variance
        Returns:
            z (torch.tensor): latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    @staticmethod
    def get_qed_reward(mol):
        """
        Calculates the QED reward for a given SMILES string
        Args:
            mol (rdkit.mol): mol file
        Returns:
            reward (float): QED score

        QED score is calculated using rdkit.Chem.QED(). Please refer to RDKit documentation.
        """
        reward = Chem.QED.qed(mol)
        return reward

    @staticmethod
    def get_fp_reward(mol, fp):
        """
        Simple metric, which calculates the fraction of input fingerprint's active bits that are
        also active in the predicted molecule's fingerprint.
        Args:
            mol (rdkit.Mol): mol of the query molecule
            fp (torch.tensor): 1-D tensor of size [4860] containing Klekota&Roth fingerprint
        Returns:
            reward (float): reward
        """
        score = 0
        key = pd.read_csv('data/KlekFP_keys.txt', header=None)
        fp_len = fp.shape[0]
        for i in range(fp_len):
            if fp[i] == 1:
                frag = Chem.MolFromSmarts(key.iloc[i].values[0])
                score += mol.HasSubstructMatch(frag)
        return score / fp_len


class EncoderDecoderV2(EncoderDecoder):
    def __init__(self, fp_size, encoding_size, hidden_size, num_layers, output_size, dropout,
                 teacher_ratio, random_seed=42, use_cuda=True):
        super().__init__(fp_size=fp_size,
                         encoding_size=encoding_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         output_size=output_size,
                         dropout=dropout,
                         teacher_ratio=teacher_ratio,
                         random_seed=random_seed,
                         use_cuda=use_cuda)
        self.fc11 = nn.Linear(self.encoding_size, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 512)
        self.relu = nn.ReLU()

    def forward(self, X, y, teacher_forcing=False, reinforcement=False):
        batch_size = X.shape[0]
        hidden = self.decoder.init_hidden(batch_size).to(self.device)
        outputs = []
        with torch.no_grad():
            mu, logvar = self.encoder(X)
            encoded = self.reparameterize(mu, logvar)
            x = encoded.unsqueeze(1)

        h1 = self.relu(self.fc11(x))
        h2 = self.relu(self.fc12(h1))
        h3 = self.relu(self.fc13(h2))
        x = h3
        # generating sequence

        for n in range(128):
            out, hidden = self.decoder(x, hidden)
            out = self.relu(self.fc1(out))  # shape (batch_size, 42)
            outputs.append(out)
            random_float = random.random()
            if teacher_forcing and random_float < self.teacher_ratio:
                out = y[:, n, :].unsqueeze(1)
            x = self.fc2(out)
        out_cat = torch.cat(outputs, dim=1)

        if reinforcement:
            rl_loss, total_reward = self.reinforce(out_cat, X)
            return out_cat, rl_loss, total_reward
        else:
            return out_cat  # out_cat.shape [batch_size, selfie_len, alphabet_len]
