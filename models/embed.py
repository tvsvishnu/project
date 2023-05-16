import torch
import torch.nn as nn

class DataEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        # Define the embedding layer
        self.embedding = nn.Linear(input_size, embedding_size)

    def forward(self, input):
        # Apply the embedding layer
        output = self.embedding(input)

        # Return the output
        return output
