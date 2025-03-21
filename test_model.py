import torch
from model import LlamaForCausalLM, LlamaConfig


config = LlamaConfig(
    vocab_size=128256,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=12,
    num_attention_heads=16
)
model = LlamaForCausalLM(config)

# Example usage
batch_size = 1
seq_length = 512
input_ids = torch.randint(0, 32000, (batch_size, seq_length))
# Create labels by shifting input_ids right by 1
labels = torch.roll(input_ids, shifts=-1, dims=1)
# Set the last token's label to -100 (ignore index)
labels[:, -1] = -100

outputs, loss = model(input_ids=input_ids, labels=labels)

print(f"Output shape: {outputs.shape}")
print(f"Loss: {loss}")
