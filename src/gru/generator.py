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
    """
    Encoder net, part of VAE.
    Parameters:
        input_size (int): size of the fingerprint vector
        output_size (int): size of the latent vectors mu and logvar
        fc1_size (int): size of the first fully connected layer
        fc2_size (int): size of the second fully connected layer
        fc3_size (int): size of the third fully connected layer
    """

    def __init__(self,
                 input_size,
                 output_size,
                 fc1_size,
                 fc2_size,
                 fc3_size):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc41 = nn.Linear(fc3_size, output_size)
        self.fc42 = nn.Linear(fc3_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.tensor): fingerprint vector
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

    @staticmethod
    def kld_loss(mu, logvar):
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return kld


class GRUDecoder(nn.Module):
    """
    Decoder class based on GRU.

    Parameters:
        hidden_size (int): GRU hidden size
        num_layers (int): GRU number of layers
        output_size (int): GRU output size (alphabet size)
        dropout (float): GRU dropout
    """

    def __init__(self, hidden_size, num_layers, output_size, dropout, input_size, encoding_size,
                 teacher_ratio, device):
        super(GRUDecoder, self).__init__()

        # GRU parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.teacher_ratio = teacher_ratio
        self.encoding_size = encoding_size
        self.output_size = output_size

        # start token initialization
        self.start_ohe = torch.zeros(42, dtype=torch.float32)
        self.start_ohe[41] = 1.0

        # pytorch.nn
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,
                          batch_first=True)
        self.fc1 = nn.Linear(self.encoding_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, latent_vector, y_true=None, teacher_forcing=False):
        """
        Args:
            latent_vector (torch.tensor): latent vector of size [batch_size, encoding_size]
            y_true (torch.tensor): batched SELFIES of target molecules
            teacher_forcing: (bool): whether to use teacher forcing (training only)

        Returns:
            out (torch.tensor): GRU output of size [batch_size, seq_len, alphabet_size]
        """
        batch_size = latent_vector.shape[0]

        # matching GRU hidden state shape
        latent_transformed = self.fc1(latent_vector)  # shape (batch_size, hidden_size)

        # initializing hidden state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        hidden[0] = latent_transformed.unsqueeze(0)  # shape (num_layers, batch_size, hidden_size)

        # initializing input (batched start token)
        x = self.start_ohe.repeat(batch_size, 1).unsqueeze(1).to(self.device)  # shape (batch_size, 1, 42)

        # generating sequence
        outputs = []
        for n in range(128):
            out, hidden = self.gru(x, hidden)
            out = (self.fc2(out))
            outputs.append(out)
            random_float = random.random()
            if (teacher_forcing and
                    random_float < self.teacher_ratio and
                    y_true is not None):
                out = y_true[:, n, :].unsqueeze(1)  # shape (batch_size, 1, 42)
            x = out
        out_cat = torch.cat(outputs, dim=1)
        return out_cat


class EncoderDecoderV3(nn.Module):
    """
    Encoder-Decoder class based on VAE and GRU. The samples from VAE latent space are passed
    to the GRU decoder as initial hidden state.

    Parameters:
        fp_size (int): size of the fingerprint vector
        encoding_size (int): size of the latent vectors mu and logvar
        hidden_size (int): GRU hidden size
        num_layers (int): GRU number of layers
        output_size (int): GRU output size (alphabet size)
        dropout (float): GRU dropout
        teacher_ratio (float): teacher forcing ratio
        random_seed (int): random seed for reproducibility
        use_cuda (bool): whether to use cuda
    """

    def __init__(self, fp_size, encoding_size, hidden_size, num_layers, output_size, dropout,
                 teacher_ratio, random_seed=42, use_cuda=True, fc1_size=2048, fc2_size=1024, fc3_size=512):
        super(EncoderDecoderV3, self).__init__()
        self.encoder = VAEEncoder(fp_size,
                                  encoding_size,
                                  fc1_size,
                                  fc2_size,
                                  fc3_size)
        self.decoder = GRUDecoder(hidden_size,
                                  num_layers,
                                  output_size,
                                  dropout,
                                  input_size=output_size,
                                  teacher_ratio=teacher_ratio,
                                  encoding_size=encoding_size,
                                  device=torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'))

        random.seed(random_seed)

    def forward(self, X, y, teacher_forcing=False, reinforcement=False, omit_encoder=False):
        """
        Args:
            X (torch.tensor): batched fingerprint vector of size [batch_size, fp_size]
            y (torch.tensor): batched SELFIES of target molecules
            teacher_forcing: (bool): whether to use teacher forcing
            reinforcement: (bool): whether to use reinforcement learning
            omit_encoder (bool): if true, the encoder is omitted and the input is passed directly to the decoder

        Returns:
            out_cat (torch.tensor): batched prediction tensor [batch_size, seq_len, alphabet_size]

        If reinforcement is enabled, the following tensors are also returned:
            rl_loss (torch.tensor): loss for use in reinforcement learning
            total_reward (torch.tensor): total reward for use in reinforcement learning
        """
        if omit_encoder:
            encoded = X
            kld_loss = torch.tensor(0.0)
        else:
            mu, logvar = self.encoder(X)
            kld_loss = self.encoder.kld_loss(mu, logvar)
            encoded = self.reparameterize(mu, logvar)  # shape (batch_size, encoding_size)

        decoded = self.decoder(latent_vector=encoded, y_true=y, teacher_forcing=teacher_forcing)
        # shape (batch_size, selfie_len, alphabet_len)

        if reinforcement:
            rl_loss, total_reward = self.reinforce(decoded, X)
            return decoded, kld_loss, rl_loss, total_reward

        else:
            return decoded, kld_loss  # out_cat.shape (batch_size, selfie_len, alphabet_len)

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
        return score / torch.sum(fp).item()
