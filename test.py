from transformer_lens import *
import json
import pprint
import argparse
import torch

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 32,
    "seq_len": 128,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
    "model_name": "bert-base-cased",
    "site": "mlp_out",
    "layer": 0,
    "device": "cuda:0",
    "normalization_type": "LN"
}
# site_to_size = {
#     "mlp_out": 512,
#     "post": 2048,
#     "resid_pre": 512,
#     "resid_mid": 512,
#     "resid_post": 512,
# }

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg

cfg = arg_parse_update_cfg(default_cfg)

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 32,
    "seq_len": 128,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
    "model_name": "bert-base-cased",
    "site": "mlp_out",
    "layer": 0,
    "device": "cuda:0",
    "normalization_type": "LN"
}
# cfg = HookedTransformerConfig(
#     n_layers=8,
#     d_model=512,
#     d_head=64,
#     n_heads=8,
#     d_mlp=2048,
#     d_vocab=61,
#     n_ctx=59,
#     act_fn="gelu",
#     normalization_type="LN",
# )

# def post_init_cfg(cfg):
#     cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
#     cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
#     cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
#     cfg["act_name"] = utils.get_act_name(cfg["site"], cfg["layer"])
#     cfg["act_size"] = site_to_size[cfg["site"]]
#     cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
#     cfg["name"] = f"{cfg['model_name']}_{cfg['layer']}_{cfg['dict_size']}_{cfg['site']}"
# post_init_cfg(cfg)
# pprint.pprint(cfg)
# %%

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# model = HookedEncoder.from_pretrained("bert-base-cased")
model = HookedTransformer.from_pretrained("distillgpt2")
# microsoft/phi-1_5
# .to(DTYPES[cfg["enc_dtype"]]).to(cfg["device"])

for name, param in model.named_parameters():
    if name.startswith("blocks.0."):
        print(name, param.shape)