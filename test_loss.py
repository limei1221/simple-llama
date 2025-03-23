import torch


if __name__ == "__main__":

    batch_size = 8
    seq_len = 128
    vocab_size = 32 * 1024

    # Generate a random logits tensor with the given shape
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Generate a random labels tensor with the given shape
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"logits.shape: {logits.shape}")
    print(f"labels.shape: {labels.shape}")

    # Calculate the loss
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

    print(f"Loss: {loss}")
