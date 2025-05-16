import torch
import torch.nn as nn

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module): #C
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x): #D
        return x
    
class DummyLayerNorm(nn.Module): #E
    def __init__(self, normalized_shape, eps=1e-5): #F
        super().__init__()
    def forward(self, x):
        return x
    
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

"""
The output tensor has two rows corresponding to the two text samples. Each
text sample consists of 4 tokens; each token is a 50,257-dimensional vector,
which matches the size of the tokenizer's vocabulary.
The embedding has 50,257 dimensions because each of these dimensions
refers to a unique token in the vocabulary. When we implement the 
postprocessing code, we will convert these 50,257-dimensional vectors back 
into token IDs, which we can then decode into words.
"""

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
batch_example = torch.randn(2, 5)
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

"""
The scale and shift are two trainable parameters (of the same dimension as the input) that
the LLM automatically adjusts during training if it is determined that doing
so would improve the model's performance on its training task. This allows
the model to learn appropriate scaling and shifting that best suit the data it is
processing. Unlike batch normalization, which normalizes across
the batch dimension, layer normalization normalizes across the feature
dimension. LLMs often require significant computational resources, and the
available hardware or the specific use case can dictate the batch size during
training or inference. Since layer normalization normalizes each input
independently of the batch size, it offers more flexibility and stability in these
scenarios. This is particularly beneficial for distributed training or when
deploying models in environments where resources are constrained.
"""

"""
GELU(x) ≈ 0.5 ⋅ x ⋅ (1 + tanh[√((2/π)) ⋅ (x + 0.044715 ⋅ x^3])
"""

class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)

    def forward(self, x):
        return self.layers(x)

"""
FeedForward module is a small
neural network consisting of two Linear layers and a GELU activation
function. In the 124 million parameter GPT model, it receives the input
batches with tokens that have an embedding size of 768 each via the
GPT_CONFIG_124M dictionary where GPT_CONFIG_124M["emb_dim"] = 768.
"""
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768) #A
out = ffn(x)
print(out.shape)

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
                            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
                      ])
    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

"""
The code implements a deep neural network with 5 layers, each consisting of
a Linear layer and a GELU activation function. In the forward pass, we
iteratively pass the input through the layers and optionally add the shortcut
connections.
"""

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0.,-1.]])
torch.manual_seed(123) # specify random seed for the initial weights for re
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])
    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
        # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean()}")

# print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)