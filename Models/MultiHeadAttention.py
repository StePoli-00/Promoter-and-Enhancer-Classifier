import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        #The initialization checks if d_model is divisible by num_heads, 
        #and then defines the transformation weights for query, key, value, and output.
                
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation layer
        self.W_k = nn.Linear(d_model, d_model) # Key transformation layer
        self.W_v = nn.Linear(d_model, d_model) # Value transformation layer 
        self.W_o = nn.Linear(d_model, d_model) # Output transformation layer
        #obviously considering the size of the input
        
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None): #here i need to apply the attention mask
        # Calculate attention scores
        # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k). 
        # Here, the attention scores are calculated by taking the dot product of queries (Q) and keys (K), 
        # and then scaling by the square root of the key dimension (d_k).
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided 
        # (useful for preventing attention to certain parts like padding)
        # this is foundamental in our specific use case since we are working with sequence with different lenght
        # so we must use the attention mask to use only the actual attention score refering the actual lenght
        if mask is not None:
            #print(mask.shape)
            #print(attn_scores.shape)
            # Reshape mask to broadcast along dimensions 2 and 3
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add two singleton dimensions

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        #This method reshapes the input x into the shape (batch_size, num_heads, seq_length, d_k). 
        # It enables the model to process multiple attention heads concurrently, allowing for parallel computation.
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        # After applying attention to each head separately, this method combines the results back into 
        # a single tensor of shape (batch_size, seq_length, d_model). This prepares the result for further processing.
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):

        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output