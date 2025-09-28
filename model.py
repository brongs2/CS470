import torch
import torch.nn as nn
import torch.optim as optim
# ✅ adjoint 쓰지 말고 일반 odeint 사용 (Hutchinson + 고계미분 섞일 때 안전)
from torchdiffeq import odeint_adjoint as odeint

# -----------------------------
# ODEfunc: 여기서는 requires_grad_ 절대 금지
# -----------------------------
class ODEfunc(nn.Module):
    def __init__(self, dim, hidden_dims):
        super().__init__()
        layers = []
        d_in = dim
        for h in hidden_dims:
            layers += [nn.Linear(d_in, h), nn.Tanh()]
            d_in = h
        layers += [nn.Linear(d_in, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, states):
        if isinstance(states, tuple):
            z, logpz = states
        else:
            z, logpz = states, None

        # Ensure z.requires_grad is True before computing dzdt
        if not z.requires_grad:
            z = z.requires_grad_()
        dzdt = self.net(z)

        if logpz is None:
            return dzdt

        # Hutchinson trace
        eps = torch.randn_like(z)
        Jv = torch.autograd.grad(dzdt, z, eps, create_graph=True)[0]
        if Jv is None:
            Jv = torch.zeros_like(z)
        div = (Jv * eps).sum(dim=1, keepdim=True)
        dlogpdt = -div
        return dzdt, dlogpdt


# -----------------------------
# CNF: 학습 경로일 때만 z.requires_grad_(True)
# -----------------------------
class CNF(nn.Module):
    def __init__(self, odefunc, method="rk4", atol=1e-3, rtol=1e-3):  # dopri5 대신 rk4, 오차 완화
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.atol = atol
        self.rtol = rtol

    def forward(self, z, logpz=None, reverse=False):
        times = torch.tensor([0.0, 1.0], device=z.device)
        if reverse:
            times = torch.flip(times, [0])

        if logpz is not None and not z.requires_grad:
            z = z.requires_grad_(True)

        state = (z, logpz) if logpz is not None else z

        out = odeint(self.odefunc, state, times,
                     atol=self.atol, rtol=self.rtol, method=self.method)

        if logpz is None:
            return out[-1], None
        else:
            z_t, logpz_t = out
            return z_t[-1], logpz_t[-1]


# -----------------------------
# CNFModel: logpz는 grad 추적 불필요(zeros), 샘플링은 logpz=None
# -----------------------------
class CNFModel(nn.Module):
    def __init__(self, dim, hidden_dims=[64, 64, 64], device="cpu"):
        super().__init__()
        self.device = device
        self.base_dist = torch.distributions.Normal(
            torch.zeros(dim, device=device),
            torch.ones(dim, device=device)
        )
        self.cnf = CNF(ODEfunc(dim, hidden_dims), method="dopri5")

    def forward(self, x):
        # ✅ logpz는 grad 추적 불필요
        logpz0 = torch.zeros(x.shape[0], 1, device=x.device)
        z, logpz_T = self.cnf(x, logpz=logpz0)
        logpz_base = self.base_dist.log_prob(z).sum(1, keepdim=True)
        return logpz_base - logpz_T  # log p(x)

    def sample(self, n):
        # ✅ 샘플링에서는 반드시 logpz=None + no_grad
        with torch.no_grad():
            z = self.base_dist.sample((n,))
            x, _ = self.cnf(z, logpz=None, reverse=True)
        return x