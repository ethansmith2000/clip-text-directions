import torch
from torch.optim import Optimizer

class NatGradLpSGD(Optimizer):
    """
    SGD with momentum where Lp-penalized parameters get natural-gradient
    preconditioning applied before momentum accumulation.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, p=0.5, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, p=p, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            precond = group["lp_precondition"]
            p = group["p"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad

                # natural-gradient preconditioner, Fisher-ish metric: multiply by |x|^(2-p)
                metric_inv = (param.abs() + eps).pow(2.0 - p)
                grad = grad * metric_inv

                state = self.state[param]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.zeros_like(param)
                else:
                    buf = state["momentum_buffer"]
                buf.mul_(mom).add_(grad)

                param.add_(buf, alpha=-lr)

        return loss
