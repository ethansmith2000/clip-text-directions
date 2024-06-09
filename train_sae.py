import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import time
from torch import nn
import torch.nn.functional as F
import json
from types import SimpleNamespace

def l0_loss_fn(x, 
            eps=1e-3, # everything below this will be considered 0
            scale = 10000.0 # push everything to [0,1] so we can count number of non-zeros
            ):
    return torch.sigmoid((x - eps) * scale).mean()


def lp_loss_fn(x, p=0.5, eps=1e-8, reduce="mean"):
    reduce = torch.mean if reduce == "mean" else torch.sum
    return (reduce(x.pow(p + eps)) + eps).pow(1/p)

# doing it in log space 
def lp_loss_fn_safe(x, p=0.5, eps=1e-8, reduce="mean"):
    reduce = torch.mean if reduce == "mean" else torch.sum
    x = torch.log(x + eps)
    x = x * p
    x = reduce(torch.exp(x))
    x = torch.log(x + eps).pow(1/p)
    return x


class ConfigurableModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = SimpleNamespace(**kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + 'config'] = vars(self.config)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if "config" in state_dict:
            _ = state_dict.pop('config')
        super().load_state_dict(state_dict, strict=strict)

    def extra_repr(self):
        return f"config={self.config}"

class BiasAdd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias

class SAE(ConfigurableModule):
    def __init__(self, 
                precond=False, #["norm", "bias"]
                mid_cond=False, #["norm", "bias"]
                out_cond=False, #["norm", "bias"]
                tied_init=False,
                dim=768, 
                expansion_factor=32,
                l1_alpha=0.1, 
                l0_alpha=0.1, 
                tc_alpha=0.1, 
                l0_eps=1e-4, 
                l0_sig_scale=10_000, 
                ):
        super().__init__(precond=precond, mid_cond=mid_cond, out_cond=out_cond, tied_init=tied_init, dim=dim, expansion_factor=expansion_factor, l1_alpha=l1_alpha, l0_alpha=l0_alpha, tc_alpha=tc_alpha, l0_eps=l0_eps, l0_sig_scale=l0_sig_scale)

        self.precond = self.mid_cond = self.out_cond = nn.Identity()
        if precond == "norm":
            self.precond = nn.LayerNorm(dim)
        elif precond == "bias":
            self.precond = BiasAdd(dim)
        
        if mid_cond == "norm":
            self.mid_cond = nn.LayerNorm(round(dim * expansion_factor))
        elif mid_cond == "bias":
            self.mid_cond = BiasAdd(round(dim * expansion_factor))
        
        if out_cond == "norm":
            self.out_cond = nn.LayerNorm(dim)
        elif out_cond == "bias":
            self.out_cond = BiasAdd(dim)

        self.up = nn.Linear(dim, round(dim * expansion_factor), bias=False)
        self.down = nn.Linear(round(dim * expansion_factor), dim, bias=False)
        self.apply(self._init_weights)
        self.register_buffer("train_step", torch.tensor(0, dtype=torch.long))
       
    def normalize_dictionary(self):
        self.down.weight.data = F.normalize(self.down.weight.data, p=2, dim=0)

    @classmethod
    def resume(cls, path):
        state_dict = torch.load(path)
        config = state_dict.pop("config")
        model = cls(**config)
        model.load_state_dict(state_dict)
        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is self.down:
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu') 
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        
        if isinstance(module, BiasAdd):
            module.bias.data.fill_(0.01)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.fill_(0.01)
            module.weight.data.fill_(0.01)

        if self.config.tied_init:
            self.down.weight.data = self.up.weight.data.T

    def encode(self, x):
        encoded = self.up(x)
        encoded_relu = F.relu(encoded)
        return encoded, encoded_relu

    def decode(self, x):
        return self.down(x)
    
    def forward(self, x):
        x = self.precond(x)
        encoded, encoded_relu = self.encode(x)
        decoder_input = self.mid_cond(encoded_relu)
        # self.normalize_dictionary()
        recon_x = self.decode(decoder_input)
        recon_x = self.out_cond(recon_x)
        self.train_step += 1

        return recon_x, encoded, encoded_relu

    
    def corr_loss(self, encoded_relu):
        encoded_relu = encoded_relu - encoded_relu.mean(dim=0)
        encoded_relu = encoded_relu / (encoded_relu.norm(dim=0) + 1e-8)
        corr = encoded_relu.T @ encoded_relu
        target = torch.eye(corr.shape[0], device=corr.device)
        return F.mse_loss(corr, target)


    def loss_fn(self, x, recon_x, encoded, encoded_relu):
        l1_loss = l0_loss = tcl = torch.zeros(1, device=x.device)
        mse_loss = F.mse_loss(recon_x, x)
        loss = mse_loss
        if self.config.l1_alpha != 0:
            l1_loss = encoded_relu.norm(p=1, dim=1).mean() / encoded_relu.shape[-1]
            loss += self.config.l1_alpha * l1_loss
        if self.config.l0_alpha != 0:
            l0_loss = l0_loss_fn(encoded_relu, eps=self.config.l0_eps, scale=self.config.l0_sig_scale)
            loss += self.config.l0_alpha * l0_loss
        if self.config.tc_alpha != 0:
            tcl = self.corr_loss(encoded)
            loss += self.config.tc_alpha * tcl

        return loss, mse_loss, l1_loss, l0_loss, tcl



def train(
    embeds,
    precond="bias",
    mid_cond="norm",
    out_cond="bias",
    tied_init=False,
    l1_alpha=0.001,
    l0_alpha=0.0,
    l0_eps=1e-3,
    l0_sig_scale=1000,
    tc_alpha=0.0,
    num_epochs=50,
    lr=8e-4,
    model_dim=768,
    expansion_factor=32,
    batch_size=8192,
    weight_decay=0.0001,
    beta1=0.9,
    beta2=0.99,
    resume_path=None,
    warmup_steps=1000,
    torch_compile=False
):
    if resume_path:
        sae = SAE.resume(resume_path).to("cuda")
    else:
        sae = SAE(precond=precond,
                mid_cond=mid_cond,
                out_cond=out_cond,
                tied_init=tied_init,
                dim=model_dim,
                expansion_factor=expansion_factor,
                l1_alpha=l1_alpha,
                l0_alpha=l0_alpha,
                tc_alpha=tc_alpha,
                l0_eps=l0_eps,
                l0_sig_scale=l0_sig_scale).to("cuda")

    sae_forward = torch.compile(sae.forward) if torch_compile else sae.forward

    dataset = TensorDataset(embeds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False)
    pbar = tqdm(range(len(dataloader) * num_epochs))

    sae_opt = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    linear1 = torch.optim.lr_scheduler.LinearLR(sae_opt, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    linear2 = torch.optim.lr_scheduler.LinearLR(sae_opt, start_factor=1.0, end_factor=0.05, total_iters=num_epochs * len(dataloader) - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=sae_opt, schedulers=[linear1, linear2], milestones=[warmup_steps])
    
    losses = {"loss": [], "mse": [], "l1": [], "l0": [], "tcl": []}
    for epoch in range(num_epochs):
        # being buggy
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False)
        for batch in dataloader:
            with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                batch = batch[0].to("cuda")
                recon_x, encoded, encoded_relu = sae_forward(batch)
                loss, mse_loss, l1_loss, l0_loss, tcl = sae.loss_fn(batch, recon_x, encoded, encoded_relu)

                sae_opt.zero_grad(set_to_none=True)
                loss.backward()
                sae_opt.step()
                scheduler.step()
                pbar.set_description(f"mse: {mse_loss.item()}, l1: {l1_loss.item()}, l0: {l0_loss.item()} tcl: {tcl.item()} epoch: {epoch}, lr: {sae_opt.param_groups[0]['lr']}")
                losses["loss"].append(loss.item())
                losses["mse"].append(mse_loss.item())
                losses["l1"].append(l1_loss.item())
                losses["l0"].append(l0_loss.item())
                losses["tcl"].append(tcl.item())
                pbar.update(1)

    torch.save(sae.state_dict(), "sae.pt")


if __name__ == "__main__":
    embeds = torch.cat(torch.load("all_embeds.pt"))
    train(embeds)