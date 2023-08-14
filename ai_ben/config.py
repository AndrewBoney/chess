from dataclasses import dataclass

@dataclass
class Config:
    input_size: int = 65
    ntokens: int = 35
    emsize: int = 200
    nhid: int = 200
    nlayers: int = 2
    nhead: int = 2
    dropout: int = 0.2