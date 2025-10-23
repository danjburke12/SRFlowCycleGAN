import torch

def checkpoint_wrapper(module, activation):
    """Wrapper function for gradient checkpointing.
    Args:
        module: PyTorch module to checkpoint
        activation: Input tensor or tuple of tensors
    Returns:
        Output of module applied to activation
    """
    def custom_forward(*inputs):
        return module(*inputs)
    
    if isinstance(activation, tuple):
        return torch.utils.checkpoint.checkpoint(custom_forward, *activation)
    return torch.utils.checkpoint.checkpoint(custom_forward, activation)
