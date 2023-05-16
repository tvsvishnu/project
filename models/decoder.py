import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()

        # Define the dimensions of the output and hidden layers
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create an LSTM layer with the specified dimensions
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)

        # Create a fully connected layer to map from the hidden state to the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Pass the input and hidden state through the LSTM layer
        output, hidden = self.lstm(x, hidden)

        # Map the hidden state to the output using the fully connected layer
        output = self.fc(output)

        # Return the output and hidden state
        return output, hidden


class DecoderLayer(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(DecoderLayer, self).__init__()

        # Create a decoder module with the specified dimensions
        self.decoder = Decoder(output_size, hidden_size, num_layers)

        # Create a layer normalization module
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, hidden):
        # Pass the input and hidden state through the decoder module
        output, hidden = self.decoder(x, hidden)

        # Normalize the output using layer normalization
        output = self.layer_norm(output)

        # Return the output and hidden state
        return output, hidden
