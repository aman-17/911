import argparse
import os
import sys
import numpy as np
from typing import List, Optional
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mup.coord_check import plot_coord_data
from mup import set_base_shapes
from mup import load_base_shapes as mup_load

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

from nn.gpt_block import muPGPTModel, muPGPTConfig
from mup_utils import save_base_shapes as local_save_base_shapes


def get_dataloader(data_paths, batch_size: int, sequence_length: int = 256):
    vocab_size = 50257
    
    class SimpleDataset:
        def __init__(self, batch_size, sequence_length, vocab_size, num_batches=100):
            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.vocab_size = vocab_size
            self.num_batches = num_batches
            
        def __iter__(self):
            for _ in range(self.num_batches):
                input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
                target_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
                yield input_ids, target_ids
                
        def __len__(self):
            return self.num_batches
    
    return SimpleDataset(batch_size, sequence_length, vocab_size)


def coord_check(
    using_mup: bool,
    widths: List,
    batch_size: int,
    nsteps: int,
    nseeds: int,
    cuda: bool = False,
    output_dir: str = "",
    load_base_shapes: Optional[str] = None,
    legend: str = "brief",
    plot: bool = True,
):
    
    def model_generator(d_model, standparam):
        def f():
            if d_model <= 64:
                n_heads = 8
            elif d_model <= 128:
                n_heads = 8
            elif d_model <= 256:
                n_heads = 16
            else:
                n_heads = max(8, (d_model // 64) * 8)
            
            while d_model % n_heads != 0:
                n_heads -= 1
                if n_heads < 1:
                    n_heads = 1
                    break
            
            base_config = muPGPTConfig.create_base_config(
                vocab_size=50257,
                max_seq_length=256,
                emb_dim=d_model,
                n_heads=n_heads,
                n_layers=12,
                drop_rate=0.1,
                qkv_bias=False,
                rope=True,
            )
            
            device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
            
            if using_mup:
                model = muPGPTModel(
                    base_config, 
                    mup_base_d_model=min(widths) if widths else d_model,
                    use_normalized_blocks=False
                )
            else:
                from nn.gpt_block import GPTModel
                model = GPTModel(base_config)
            
            model = model.to(device)
            return model
        return f

    optimizer = "adamw"
    
    models = {width: model_generator(width, standparam=not using_mup) for width in widths}
    model_instances = {width: model_fn() for width, model_fn in models.items()}

    if using_mup:
        print("Setting up muP base shapes...")
        
        if load_base_shapes and os.path.exists(load_base_shapes):
            print(f"Loading base shapes from {load_base_shapes}")
            base_shapes = mup_load(load_base_shapes)
            for model in model_instances.values():
                set_base_shapes(model, base_shapes)
        
        else:
            base_width = min(widths)
            base_model_fn = model_generator(base_width, standparam=False)
            base_model = base_model_fn()
            
            larger_width = max(widths) if len(widths) > 1 else base_width * 2
            larger_model_fn = model_generator(larger_width, standparam=False)
            larger_model = larger_model_fn()
            
            from mup import get_shapes
            base_shapes_dict = get_shapes(base_model)
            delta_shapes_dict = get_shapes(larger_model)
            
            for model in model_instances.values():
                set_base_shapes(model, base_shapes_dict, delta_shapes_dict)
        
            del base_model, larger_model
    
    data_loader = get_dataloader(data_paths='dummy', batch_size=batch_size)
            
    from coord_check import get_coord_data
    
    def loss_fn(model, batch):
        input_batch, target_batch = batch
        device = next(model.parameters()).device
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_batch.view(-1)
        )
        return loss
    
    df = get_coord_data(
        models,
        dataloader=data_loader,
        load_base_shapes=None, 
        mup=using_mup,
        lr=1e-3,
        optimizer=optimizer,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn=loss_fn,
        cuda=cuda,
    )

    prm = "mup" if using_mup else "sp"
    os.makedirs(output_dir, exist_ok=True)
    coords_file = os.path.join(output_dir, f"{prm}_gpt_{optimizer}_coord.csv")
    df.to_csv(coords_file, index=False)
    
    if plot:
        step_interval = max(nsteps // 20, 1)
        df = df[df["t"] % step_interval == 0]
        df.loc[:, "t"] /= step_interval

        plot_coord_data(
            df,
            legend=legend,
            save_to=os.path.join(output_dir, f"{prm}_gpt_{optimizer}_coord.png"),
            suptitle=f"{prm} Transformer {optimizer} nseeds={nseeds}",
            face_color="xkcd:light grey" if not using_mup else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coord check for GPT model with muP",
    )

    parser.add_argument("--save_base_shapes", type=str, default="", help="file location to save base shapes at")
    parser.add_argument("--load_base_shapes", type=str, default="", help="file location to load base shapes from")

    parser.add_argument("--batch_size", type=int, default=8, metavar="N", help="batch size")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512], help="widths to use for coord check")

    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--legend", type=str, default="brief", help="'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.")

    parser.add_argument(
        "--coord_check",
        action="store_true",
        help="test mup is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.",
    )
    parser.add_argument("--coord_check_nsteps", type=int, default=3, help="Do coord check with this many steps.")
    parser.add_argument(
        "--coord_check_nseeds",
        type=int,
        default=3,
        help="number of seeds for testing correctness of mup",
    )

    parser.add_argument(
        "--coord_check_save_path",
        type=str,
        default="coord_checks",
        help="dir location for saving coord check plots",
    )

    args = parser.parse_args()
    print(args)

    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.save_base_shapes:
        local_save_base_shapes(args.save_base_shapes, d_model=64)
        print("Base shapes saved and exit")
        import sys
        sys.exit()

    if args.coord_check:
        print("testing parametrization")

        os.makedirs(args.coord_check_save_path, exist_ok=True)

        for mup_usage in [True, False]:
            coord_check(
                using_mup=mup_usage,
                widths=args.widths,
                batch_size=args.batch_size,
                nsteps=args.coord_check_nsteps,
                nseeds=args.coord_check_nseeds,
                cuda=args.cuda,
                output_dir=args.coord_check_save_path,
                legend=args.legend,
                load_base_shapes=args.load_base_shapes,
            )
