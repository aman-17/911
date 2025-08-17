import torch
import torch.nn as nn
import inspect

class Optimizer911(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def _get_device_type(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def configure_optimizers(self, model, weight_decay, learning_rate, betas, device_type):
            param_dict = {pn: p for pn, p in model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            if self.cfg.mup:
                mup_decay_params = []
                decay_params = []
                nodecay_params = []
                for n, p in param_dict.items():
                    if p.dim() >= 2:
                        if n.endswith('c_attn.weight') or n.endswith('c_fc.weight') or n.endswith('c_proj.weight'):
                            mup_decay_params.append(p)
                        else:
                            decay_params.append(p)
                    else:
                        nodecay_params.append(p)
                optim_groups = [
                    {'params': mup_decay_params, 'weight_decay': weight_decay, 'lr_scale': 1/self.cfg.mup_width_multiplier},
                    {'params': decay_params, 'weight_decay': weight_decay, 'lr_scale': 1},
                    {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1}
                ]
                num_mup_decay_params = sum(p.numel() for p in mup_decay_params)
                num_decay_params = sum(p.numel() for p in decay_params)
                num_nodecay_params = sum(p.numel() for p in nodecay_params)
                print(f"num mup decayed parameter tensors: {len(mup_decay_params)}, with {num_mup_decay_params:,} parameters")
                print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

            else:
                decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
                nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
                optim_groups = [
                    {'params': decay_params, 'weight_decay': weight_decay},
                    {'params': nodecay_params, 'weight_decay': 0.0}
                ]
                num_decay_params = sum(p.numel() for p in decay_params)
                num_nodecay_params = sum(p.numel() for p in nodecay_params)
                print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            
            if self.cfg.mup:
                for group in optim_groups:
                    group['lr'] = learning_rate * group.pop('lr_scale')
            
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")

            return optimizer