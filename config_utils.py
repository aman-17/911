from typing import Any, Dict

import yaml


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"


def print_config_value(key: str, value: Any, indent: int = 4) -> None:

    spaces = " " * indent
    colored_key = f"{Colors.CYAN}{key}{Colors.RESET}"
    if isinstance(value, str):
        colored_value = f"{Colors.GREEN}'{value}'{Colors.RESET}"
    elif isinstance(value, bool):
        colored_value = f"{Colors.YELLOW}{value}{Colors.RESET}"
    elif isinstance(value, (int, float)):
        colored_value = f"{Colors.MAGENTA}{value}{Colors.RESET}"
    elif value is None:
        colored_value = f"{Colors.DIM}None{Colors.RESET}"
    else:
        colored_value = f"{Colors.WHITE}{value}{Colors.RESET}"

    print(f"{spaces}{colored_key}={colored_value},")


def print_configuration(config: Dict[str, Any], variant_name: str) -> None:

    print(f"\n{Colors.BOLD}{Colors.BG_BLUE} MODEL CONFIGURATION {Colors.RESET}")
    print(f"{Colors.BOLD}TrainingConfig{Colors.RESET}(")

    print(
        f"    {Colors.BOLD}{Colors.BLUE}model={Colors.RESET}{Colors.YELLOW}ModelConfig{Colors.RESET}("
    )
    model_keys = [
        "type",
        "model_arch",
        "emb_dim",
        "n_heads",
        "n_layers",
        "hidden_dim",
        "n_kv_heads",
    ]
    for key in model_keys:
        if key in config:
            print_config_value(key, config[key], 8)
    print("    ),")

    print(
        f"    {Colors.BOLD}{Colors.BLUE}training={Colors.RESET}{Colors.YELLOW}TrainingConfig{Colors.RESET}("
    )
    training_keys = [
        "batch_size",
        "num_epochs",
        "initial_lr",
        "weight_decay",
        "drop_rate",
    ]
    for key in training_keys:
        if key in config:
            print_config_value(key, config[key], 8)
    print("    ),")

    print(
        f"    {Colors.BOLD}{Colors.BLUE}dataset={Colors.RESET}{Colors.YELLOW}DatasetConfig{Colors.RESET}("
    )
    data_keys = [
        "train_data",
        "vocab_size",
        "max_seq_length",
        "seq_length",
        "stride",
        "num_workers",
        "pin_memory",
        "shuffle_buffer_size",
    ]
    for key in data_keys:
        if key in config:
            print_config_value(key, config[key], 8)
    print("    ),")

    print(
        f"    {Colors.BOLD}{Colors.BLUE}attention={Colors.RESET}{Colors.YELLOW}AttentionConfig{Colors.RESET}("
    )
    attention_keys = [
        "attention",
        "use_flash_attn",
        "qkv_bias",
        "rope",
        "tie_embeddings",
        "dtype",
        "softcap",
        "query_pre_attn_scalar",
    ]
    for key in attention_keys:
        if key in config:
            print_config_value(key, config[key], 8)
    print("    ),")

    rope_keys = [
        "rope_theta",
        "rope_factor",
        "qk_rope_head_dim",
        "qk_nope_head_dim",
        "v_head_dim",
        "beta_fast",
        "beta_slow",
    ]
    if any(key in config for key in rope_keys):
        print(
            f"    {Colors.BOLD}{Colors.BLUE}rope={Colors.RESET}{Colors.YELLOW}RoPEConfig{Colors.RESET}("
        )
        for key in rope_keys:
            if key in config:
                print_config_value(key, config[key], 8)
        print("    ),")

    mla_keys = [
        "compression_block_size",
        "compression_stride",
        "selection_block_size",
        "selection_top_k",
        "window_size",
        "q_lora_rank",
        "kv_lora_rank",
    ]
    if any(key in config for key in mla_keys):
        print(
            f"    {Colors.BOLD}{Colors.BLUE}mla={Colors.RESET}{Colors.YELLOW}MLAConfig{Colors.RESET}("
        )
        for key in mla_keys:
            if key in config and config[key] is not None:
                print_config_value(key, config[key], 8)
        print("    ),")

    print(")")
    if all(key in config for key in ["emb_dim", "n_layers", "vocab_size"]):
        print(f"\n{Colors.BOLD}{Colors.GREEN}Model Statistics:{Colors.RESET}")
        emb_params = config["vocab_size"] * config["emb_dim"]
        if not config.get("tie_embeddings", False):
            emb_params *= 2

        if config.get("hidden_dim"):
            block_params = config["n_layers"] * (
                4 * config["emb_dim"] * config["emb_dim"]
                + 2 * config["emb_dim"] * config["hidden_dim"]
            )
        else:
            block_params = (
                config["n_layers"] * 12 * config["emb_dim"] * config["emb_dim"]
            )

        total_params = emb_params + block_params
        non_emb_params = total_params - emb_params

        print(
            f"{Colors.CYAN}Total parameters:{Colors.RESET}         {Colors.MAGENTA}{total_params:,}{Colors.RESET} ({Colors.MAGENTA}{total_params:,}{Colors.RESET} active)"
        )
        print(
            f"{Colors.CYAN}Non-embedding parameters:{Colors.RESET} {Colors.MAGENTA}{non_emb_params:,}{Colors.RESET} ({Colors.MAGENTA}{non_emb_params:,}{Colors.RESET} active)"
        )
        print(
            f"{Colors.CYAN}Active variant:{Colors.RESET}           {Colors.GREEN}'{variant_name}'{Colors.RESET}"
        )


def load_config(config_path: str = "config.yaml", model_variant: str = None):
    """Load configuration and extract the specified model variant.

    Args:
        config_path: Path to the config file
        model_variant: Specific model variant to use (if None, uses active)

    Returns:
        dict: Flattened configuration ready for model initialization
    """
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    if model_variant is None:
        active_variant = full_config["model"]["active"]
    else:
        active_variant = model_variant

    if active_variant not in full_config["model"]["variants"]:
        available = list(full_config["model"]["variants"].keys())
        raise ValueError(
            f"Model variant '{active_variant}' not found. Available: {available}"
        )

    model_config = full_config["model"]["variants"][active_variant]
    train_config = {**full_config, **model_config}
    train_config.pop("model", None)
    train_config["active_variant"] = active_variant
    if train_config.get("print_config", False):
        print_configuration(train_config, active_variant)

    return train_config


def list_model_variants(config_path: str = "config.yaml"):
    """List all available model variants.

    Returns:
        list: Available model variant names
    """
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    return list(full_config["model"]["variants"].keys())
