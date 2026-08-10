import os

import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        self._ensure_capacity(prev_length + tgt.shape[dim], dim=dim)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        start = int(self.current_length.item())
        needed = start + tensor.shape[dim]
        self._ensure_capacity(needed, dim=dim)
        dst = self.data.narrow(dim, start, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, int(self.current_length.item()))

    def _ensure_capacity(self, needed_length: int, dim: int = 2):
        """Grow cache tensor on demand when preallocated context is insufficient."""
        current_capacity = self.data.shape[dim]
        if needed_length <= current_capacity:
            return

        # Grow geometrically to avoid frequent reallocations on long generations.
        new_capacity = max(needed_length, current_capacity + max(current_capacity // 2, 256))
        pad_shape = list(self.data.shape)
        pad_shape[dim] = new_capacity - current_capacity
        pad = torch.zeros(
            *pad_shape,
            device=self.data.device,
            dtype=self.data.dtype,
        )
        self.data = torch.cat((self.data, pad), dim=dim)


def resolve_initial_cache_capacity(config):
    max_position_embeddings = int(getattr(config, "max_position_embeddings", 4096))
    raw_capacity = os.environ.get("MMSPEC_LONG_CONTEXT_INITIAL_KV_CACHE_CAPACITY", "4096")
    if raw_capacity in ("", "none", "None", "0"):
        return max_position_embeddings
    try:
        capacity = int(raw_capacity)
    except ValueError:
        capacity = 4096
    if capacity <= 0:
        return max_position_embeddings
    return min(max_position_embeddings, capacity)


def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # For multimodal wrappers (e.g., LLaVA), cache lives in the language model.
    if hasattr(model, "language_model"):
        cache_model = model.language_model
    else:
        cache_model = model

    # Extracting configuration from the decoder/language model.
    config = cache_model.config
    if not hasattr(config, "num_hidden_layers") and hasattr(model.config, "text_config"):
        config = model.config.text_config
    cache_capacity = resolve_initial_cache_capacity(config)
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    devices = []
    for i in range(config.num_hidden_layers):
        try:
            device = cache_model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device = cache_model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    cache_dtype = getattr(cache_model, "dtype", model.dtype)
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]
    for id, i in enumerate(devices):
        assert i.type == "cuda"  # TODO

        if startdevice != i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                cache_capacity,
                config.hidden_size // config.num_attention_heads,
                device=startdevice,
                dtype=cache_dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum = 0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        cache_capacity,
        config.hidden_size // config.num_attention_heads,
        device=startdevice,
        dtype=cache_dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers

    bias = 0
    start_data_m = devices[0].index
    for i in range(config.num_hidden_layers):
        data_m = devices[i].index
        if data_m != start_data_m:
            bias = 0
            start_data_m = data_m
        try:
            past_key_values.append(
                [
                    KVCache(
                        past_key_values_data_list[data_m - devices[0].index][
                            2 * bias + j
                        ],
                        current_length_data[i * 2 + j],
                    )
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(
                        past_key_values_data_list[0][2 * bias + j],
                        current_length_data[i * 2 + j],
                    )
                    for j in range(2)
                ]
            )
        bias += 1
    return past_key_values, past_key_values_data_list, current_length_data
