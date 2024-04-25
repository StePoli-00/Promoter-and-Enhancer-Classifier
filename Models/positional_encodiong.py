import torch
import torch.nn as nn 
import math



class PositionalEncoding(nn.Module): #subclass of nn.Module allowing it to be used as a pytorch layer
    def __init__(self, d_model, max_seq_length):
        #d_model, the dimension of the model's input 
        #max_seq_lenght, maximun lenght of the input sequence 
        super(PositionalEncoding, self).__init__()
        
        #tensor setted equal to zero that will be populated with pos encodings
        pe = torch.zeros(max_seq_length, d_model) 
        
        #torch.arange create a tensor of index from 0 up to max lenght
        #with unsqueeze we add a dimension so the tensor will pass from [max_len] to [max_len, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        #A term used to scale the position indices in a specific way.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) #sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term) #cosine to the odd indices
        
        self.register_buffer('pe', pe.unsqueeze(0)) 
        
        
    def forward(self, x):
        #The forward method simply adds the positional encodings to the input x.

        #It uses the first x.size(1) elements of pe to ensure that the positional 
        # encodings match the actual sequence length of x.
        
        return x + self.pe[:, :x.size(1)]