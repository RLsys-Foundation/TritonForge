from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import torch


@dataclass
class Sample:
    """The sample generated"""

    index: Optional[int] = None
    # prompt
    prompt: Union[str, list[dict[str, str]]] = ""
    tokens: list[int] = field(default_factory=list)
    # response
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, dict[str, float]]] = None
    loss_mask: Optional[list[int]] = None

    # Multi-turn support
    turn_idx: int = 0  # Current turn index (0-based)
    history: list[dict] = field(default_factory=list)  # Previous turns with code + eval results
    turn_rewards: list[float] = field(default_factory=list)  # Rewards for each turn
    aggregated_return: Optional[float] = None  # Discounted aggregated return

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data["status"] = Sample.Status(data["status"])
        return Sample(**data)


@dataclass
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int
