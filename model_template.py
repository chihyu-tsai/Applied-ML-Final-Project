import torch
from torch import nn
from typing import Any, Iterable, List

from pathlib import Path

CKPT_PATH = Path(__file__).resolve().parent / "model.pt"
_ckpt = torch.load(CKPT_PATH, map_location="cpu")
PAD_id = _ckpt["PAD_id"]
state_dict = _ckpt["model_state_dict"]

# read embedding size from saved weights
emb_weight = state_dict["embedding.weight"]
vocab_size, d_embed = emb_weight.shape

# RNN classifier
class RNN_Classifier(nn.Module):

    def __init__(self,
                 n_embed,
                 d_embed,  # for GloVe this is 300 - dim per token
                 d_hidden,
                 d_out=2,
                 embeddings=None,
                 num_layers=2,
                 bidirectional=True,
                 rnn_type="LSTM",
                 dr=0.5):

        super().__init__()
        self.num_embeddings = n_embed
        self.d_embed = embeddings.shape[1] if embeddings is not None else d_embed
        self.hidden_dim = d_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = dr
        self.rnn_type = rnn_type

        # embedding layer
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings, freeze=True, padding_idx=PAD_id
            )
        else:
            self.embedding = nn.Embedding(n_embed, d_embed)

        # rnn layer
        if rnn_type == "RNN":  # simple RNN
            self.rnn = nn.RNN(
                self.d_embed,
                self.hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=self.dropout,
            )
        elif rnn_type == "GRU":  # GRU
            self.rnn = nn.GRU(
                self.d_embed,
                self.hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=self.dropout,
            )
        else:  # LSTM
            self.rnn = nn.LSTM(
                self.d_embed,
                self.hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=self.dropout,
            )

        effective_hidden = d_hidden * (2 if self.bidirectional else 1)
        # fully connected layer
        self.fc = nn.Linear(effective_hidden, d_out)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, text):
        # text: (batch_size, max_len)
        embedded = self.embedding(text)  # -> (batch_size, max_len, d_embed)

        output, hidden = self.rnn(embedded)  # shape depends on RNN type

        if self.rnn_type == "LSTM":
            hidden = hidden[0]  # take h_n, ignore c_n

        num_dirs = 2 if self.bidirectional else 1
        h_n = hidden.view(self.num_layers, num_dirs, output.shape[0], self.hidden_dim)

        last_hidden = h_n[-1]  # last layer

        if self.bidirectional:
            last_hidden_forward, last_hidden_backward = last_hidden[0], last_hidden[1]
            h_cat = torch.cat((last_hidden_forward, last_hidden_backward), dim=-1)
            last_hidden = h_cat
        else:
            last_hidden = last_hidden[0]

        last_hidden = self.dropout_layer(last_hidden)
        logits = self.fc(last_hidden)  # (batch_size, d_out)

        return logits


class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize your model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dummy_emb = torch.zeros((vocab_size, d_embed), dtype=torch.float32)
        self.net = RNN_Classifier(
            n_embed=vocab_size,
            d_embed=d_embed,
            d_hidden=64,
            d_out=2,
            embeddings=dummy_emb,
            num_layers=2,
            bidirectional=True,
            rnn_type="LSTM",
            dr=0.5,
        ).to(self.device)

        self.net.load_state_dict(state_dict)

    def eval(self) -> None:
        # Optional: set your model to evaluation mode
        super().eval()
        self.net.eval()
        return None

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        self.eval()
        if isinstance(batch, torch.Tensor):
            Xb = batch.to(self.device)
        else:
            Xb = torch.stack(list(batch)).to(self.device)

        logits = self.net(Xb)
        preds = logits.argmax(dim=-1)
        return preds.cpu().tolist()
        # raise NotImplementedError("Implement predict(...) to return a list of labels.")


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()


