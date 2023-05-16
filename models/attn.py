import torch
import torch.nn as nn

class FullAttention(nn.Module):
    def forward(self, query, key, value):
        # Compute the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale the scores by the square root of the key dimension
        scores = scores / torch.sqrt(torch.tensor(key.shape[-1]).float())

        # Apply the softmax function to get the attention weights
        weights = torch.softmax(scores, dim=-1)

        # Compute the context vector as a weighted sum of the value vectors
        context = torch.matmul(weights, value)

        # Return the context vector and the attention weights
        return context, weights


class ProbAttention(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Compute the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale the scores by the square root of the key dimension
        scores = scores / torch.sqrt(torch.tensor(key.shape[-1]).float())

        # Apply the softmax function to get the attention weights
        weights = torch.softmax(scores, dim=-1)
        print("Weight Matrix : ", weights)
        print("Length : ", len(weights))
        # Apply dropout to the attention weights
        weights = self.dropout(weights)
        print("Weight Matrix : ", weights)
        print("Length : ", len(weights))
        # Compute the context vector as a weighted sum of the value vectors
        context = torch.matmul(weights, value)

        # Return the context vector and the attention weights
        return context, weights


class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1, attention_type='prob'):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Define the query, key, and value transformations
        self.query_transform = nn.Linear(input_size, output_size, bias=False)
        self.key_transform = nn.Linear(input_size, output_size, bias=False)
        self.value_transform = nn.Linear(input_size, output_size, bias=False)

        # Define the attention mechanism
        if attention_type == 'full':
            self.attention = FullAttention()
        elif attention_type == 'prob':
            self.attention = ProbAttention(dropout=dropout)

        # Define the output transformation
        self.output_transform = nn.Linear(output_size, output_size, bias=False)

        # Define the dropout module
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, context):
        # Compute the queries, keys, and values
        query = self.query_transform(input)
        key = self.key_transform(context)
        value = self.value_transform(context)

        # Compute the context vector and the attention weights
        context, weights = self.attention(query, key, value)

        # Apply dropout to the context vector
        context = self.dropout(context)

        # Apply the output transformation
        output = self.output_transform(context)

        # Return the output and the attention weights
        return output, weights
