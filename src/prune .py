import torch
import torch.nn.utils.prune as prune

def prune_weights(model, amount=0.3):
    '''
    Prunes the model by removing low-magnitude weights.
    
    Parameters:
        model (torch.nn.Module): The model to be pruned.
        amount (float): Fraction of connections to prune.
    '''
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent

def apply_pruning(model, amount=0.3):
    '''
    Applies pruning and returns the pruned model.
    '''
    prune_weights(model, amount)
    return model
