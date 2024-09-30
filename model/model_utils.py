import numpy as np


def is_all_zero(audio_array, threshold=0.1):
    # Function to check if an audio sample starts with all zeros
    return np.all(audio_array[:int(threshold * len(audio_array))] == 0)


def get_eeg_embedding_hook(model):
    """Registers a hook on the conv_separable_point layer and returns a function to retrieve its output."""

    # Placeholder for the feature map
    conv_output = None

    # Define the hook function
    def hook_fn(module, input, output):
        nonlocal conv_output
        conv_output = output

    # Register the hook to the conv_separable_point layer
    target_layer = model.pool_2
    hook_handle = target_layer.register_forward_hook(hook_fn)

    def get_output():
        """Function to get the conv_separable_point layer's output."""
        return conv_output

    return get_output, hook_handle


def get_aud_embedding_hook(model):
    """Registers a hook on the conv_separable_point layer and returns a function to retrieve its output."""

    # Placeholder for the feature map
    conv_output = None

    # Define the hook function
    def hook_fn(module, input, output):
        nonlocal conv_output
        conv_output = output

    # Register the hook to the conv_separable_point layer
    target_layer = model.projector
    hook_handle = target_layer.register_forward_hook(hook_fn)

    def get_output():
        """Function to get the conv_separable_point layer's output."""
        return conv_output

    return get_output, hook_handle


def get_tou_embedding_hook(model):
    """Registers a hook on the conv_separable_point layer and returns a function to retrieve its output."""

    # Placeholder for the feature map
    conv_output = None

    # Define the hook function
    def hook_fn(module, input, output):
        nonlocal conv_output
        conv_output = output

    # Register the hook to the conv_separable_point layer
    target_layer = model.layers[1]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    def get_output():
        """Function to get the conv_separable_point layer's output."""
        return conv_output

    return get_output, hook_handle

