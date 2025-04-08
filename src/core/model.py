import igraph
import readdy
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
import igraph as ig

from src.core.model_utils import ModelUtils

# TODO: Set up metadata to store info on how the model was constructed
@dataclass
class Model(ModelUtils):
    topology_type: Optional[str] = field(default=None, repr=True)
    particle_type: Optional[str] = field(default=None, repr=True)
    parameters: Optional[dict] = field(default=None, repr=False)
    data: Optional[any] = field(default=None, repr=False)
    display_properties: Optional[dict] = field(default=None, repr=False)

    def __len__(self):
        if self.data is None:
            return 0
        if isinstance(self.data, ig.Graph):
                self.data = [self.data]
        if isinstance(self.data, list):
            n = 0
            for g in self.data:
                n += len(g.vs)
            return n
        else:
            return self.data.shape[0]

