# coding=utf-8

import dataclasses
from dataclasses import dataclass
import json


@dataclass
class VqvaeConfig:
    in_channel: int = 3
    channel: int = 128
    n_res_block: int = 2
    n_res_channel: int = 32
    embed_dim: int = 64
    n_embed: int = 512
    decay: float = 0.99
    bottom_stride: int = 4

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))


@dataclass
class VqVaeConfig2:
    bottom_embed_dim: int = 64
    bottom_n_embed: int = 1024
    top_embed_dim: int = 64
    top_n_embed: int = 256

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))
