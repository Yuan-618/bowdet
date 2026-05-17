"""MERT-based model for bow change detection"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, input_values):
        outputs = self.mert(input_values, output_hidden_states=False)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden).squeeze(1)
