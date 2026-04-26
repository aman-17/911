import torch

from interpretability.models.olmo2_1b import load_model
from interpretability.nn.sae import SAEConfig, SAEOutput, SparseAutoencoder


class FeatureSteerer:
    """Steers model activations by scaling SAE features at a given layer.

    Usage:
        steerer = FeatureSteerer(model, sae, layer_idx=8)
        with steerer.set_feature(42, scale=3.0):
            output = model.generate(...)
    """

    def __init__(self, model: torch.nn.Module, sae: SparseAutoencoder, layer_idx: int) -> None:
        self._model = model
        self._sae = sae
        self._layer_idx = layer_idx
        self._scales: dict[int, float] = {}
        self._handle = None

    def set_feature(self, feature_idx: int, scale: float) -> "FeatureSteerer":
        self._scales[feature_idx] = scale
        return self

    def _hook(self, _module: torch.nn.Module, _input: tuple, output: torch.Tensor | tuple) -> torch.Tensor | tuple:
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            out: SAEOutput = self._sae(hidden)
            features = out.features.clone()
            for idx, scale in self._scales.items():
                features[..., idx] *= scale
            steered = self._sae.decode(features) + (hidden - out.recon)
        return (steered,) if isinstance(output, tuple) else steered

    def __enter__(self) -> "FeatureSteerer":
        layer = self._model.model.layers[self._layer_idx]
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            self._handle.remove()


def run_steered_generation(
    feature_idx: int,
    scale: float = 3.0,
    prompt: str = "Hello!",
    max_new_tokens: int = 200,
    layer_idx: int = 8,
    checkpoint: str = "olmo2_1b_sae_layer8.pt",
) -> str:
    model, tokenizer = load_model()
    model.eval()

    sae = SparseAutoencoder(SAEConfig()).to(model.device)
    sae.load_state_dict(torch.load(checkpoint, map_location=model.device, weights_only=True))
    sae.eval()

    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with FeatureSteerer(model, sae, layer_idx).set_feature(feature_idx, scale):
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(run_steered_generation(feature_idx=0, scale=3.0))
