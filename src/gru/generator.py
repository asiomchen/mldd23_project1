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
        self.sigmoid = nn.Sigmoid()

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
        logvar = self.fc41(h3)
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
            reinforcement (bool):enable loss calculation for use
                                 in reinforcement learning
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
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

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
    def __init__(self, fp_size=4860, encoding_size=512, hidden_size=512, num_layers=0, output_size=42, dropout=0,
                 teacher_ratio=0.5, random_seed=42):
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
        """
        super(EncoderDecoder, self).__init__()
        self.teacher_ratio = teacher_ratio
        self.encoder = VAEEncoder(fp_size, encoding_size)
        self.decoder = DecoderNet(encoding_size, hidden_size, num_layers, output_size, dropout)
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        random.seed(random_seed)

        # pytorch.nn
        self.fc1 = nn.Linear(hidden_size, 42)
        self.fc2 = nn.Linear(42, hidden_size)
        self.relu = nn.ReLU()
        self.softmax2d = nn.Softmax(dim=2)

    def forward(self, x, y, teacher_forcing=False, reinforcement=False):
        """
        Args:
            x (torch.tensor):batched fingerprint vector of size [batch_size, fp_size]
            y (torch.tensor):batched SMILES of target molecules
            teacher_forcing: (bool):enable teacher forcing
            reinforcement: (bool):enable loss calculation for use in reinforcement learning

        Returns:
            out_cat (torch.tensor):batched vectorized SELFIES tensor
                                   of size [batch_size, seq_len, alphabet_size]

        If reinforcement is enabled, the following tensors are also returned:
            rl_loss (torch.tensor):loss for use in reinforcement learning
            total_reward (torch.tensor):total reward for use in reinforcement learning
        """
        batch_size = x.shape[0]
        hidden = self.decoder.init_hidden(batch_size).to(self.device)
        mu, logvar = self.encoder(x)
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
            x = self.relu(self.fc2(out))
        out_cat = torch.cat(outputs, dim=1)
        # out_cat = self.softmax2d(out_cat)

        if reinforcement:
            rl_loss, total_reward = self.reinforce(out_cat)
            return out_cat, rl_loss, total_reward
        else:
            return out_cat  # out_cat.shape [batch_size, selfie_len, alphabet_len]

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
        z = eps.mul(std).add_(mu)
        return z

    def reinforce(self, out_cat, n_samples=10):
        """
        Reinforcement learning loop for use in training the model.
        Args:
            out_cat (torch.tensor):batched vectorized SELFIES tensor
            n_samples (int):number of samples to draw from batch
        Returns:
            rl_loss (torch.tensor):loss for use in reinforcement learning
            total_reward (torch.tensor):total reward for use in reinforcement learning
        """
        vectorizer = SELFIESVectorizer(pad_to_len=128)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_cat = out_cat.detach().cpu().numpy()

        # reinforcement learning    
        rl_loss = 0
        total_reward = 0
        gamma = 0.97
        batch_size = out_cat.shape[0]
        total_reward = 0

        sample_idxs = [random.randint(0, batch_size - 1) for x in range(n_samples)]
        valid = False
        sf.set_semantic_constraints("hypervalent")
        # check if able to decode all, else reroll
        while not valid:
            for idx in sample_idxs:
                try:
                    seq = vectorizer.devectorize(out_cat[idx], remove_special=True)
                    _ = sf.decoder(seq)
                    valid = True
                except:
                    sample_idxs = [random.randint(0, batch_size - 1) for x in range(n_samples)]
                    valid = False
                    print('Exception caught, rerolling')
                    break
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
            for p in range(len(trajectory) - 1):
                token = trajectory_input[p]
                token_idx = torch.argmax(token.detach().cpu()).item()
                token = token.float().to(device)
                while token_idx != 40 and token_idx != 39:  # until encounters [nop] or [end]
                    representation = self.relu(self.fc2(token)).unsqueeze(0)  # [1, 512]
                    representation, hidden = self.decoder(representation, hidden)
                    representation = representation.squeeze(0)  # [512]
                    next_token = self.relu(self.fc1(representation))  # [42]

                    log_probs = F.log_softmax(next_token, dim=0)  # [42]
                    top_i = trajectory_input[p + 1].long()
                    rl_loss -= (log_probs[top_i] * discounted_reward)
                    discounted_reward = discounted_reward * gamma

        rl_loss = rl_loss / n_samples
        total_reward = total_reward / n_samples
        return rl_loss, total_reward

    @staticmethod
    def get_reward(smiles):
        """
        Calculates the reward for a given SMILES string
        Args:
            smiles: SMILES string
        Returns:
            reward: float
        """
        mol = Chem.MolFromSmiles(smiles)
        reward = Chem.QED.qed(mol)
        return reward