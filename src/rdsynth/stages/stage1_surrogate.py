from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rdsynth.models.mlp import MLP


@dataclass
class SurrogateBundle:
    surrogate: nn.Module
    generator: nn.Module
    query_count: int = 0
    runtime_sec: float = 0.0
    round_log: list[dict[str, float]] | None = None


def train_surrogate(
    oracle,
    feature_dim: int,
    n_classes: int,
    z_dim: int,
    gen_hidden,
    sur_hidden,
    steps: int,
    batch_size: int,
    lr_s: float,
    lr_g: float,
    device: torch.device,
    log_every: int = 200,
    query_budget: int | None = None,
    consistency_weight: float = 0.0,
    consistency_noise: float = 0.0,
    real_x: torch.Tensor | None = None,
    real_y: torch.Tensor | None = None,
    real_data_ratio: float = 0.0,
    class_weight: str | None = None,
):
    generator = MLP(z_dim, gen_hidden, feature_dim).to(device)
    surrogate = MLP(feature_dim, sur_hidden, n_classes).to(device)

    opt_s = torch.optim.Adam(surrogate.parameters(), lr=lr_s)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g)

    weight_tensor = None
    if class_weight == "balanced" and real_y is not None:
        counts = torch.bincount(real_y.cpu(), minlength=n_classes).float()
        raw_weights = counts.sum() / (counts + 1.0e-6)
        raw_weights = torch.where(counts > 0, raw_weights, torch.ones_like(raw_weights))
        weight_tensor = raw_weights.to(device)
    if class_weight is not None and weight_tensor is None:
        weight_tensor = torch.ones(n_classes, device=device)
    ce = nn.CrossEntropyLoss(weight=weight_tensor)

    queries = 0
    loss_g = torch.tensor(0.0, device=device)
    if real_x is not None:
        real_x = real_x.to(device)
    if real_y is not None:
        real_y = real_y.to(device)
    for step in range(1, steps + 1):
        use_real = real_x is not None and real_y is not None and real_data_ratio > 0
        n_real = int(batch_size * real_data_ratio) if use_real else 0
        n_gen = batch_size - n_real

        loss_s = torch.tensor(0.0, device=device)
        if n_gen > 0:
            z = torch.randn(n_gen, z_dim, device=device)
            x_gen = generator(z)
            if query_budget is None or queries < query_budget:
                with torch.no_grad():
                    y_gen = oracle.predict(x_gen)
                queries += n_gen
                logits = surrogate(x_gen)
                loss_s = loss_s + ce(logits, y_gen)
            else:
                logits = surrogate(x_gen)
                loss_s = loss_s + torch.tensor(0.0, device=device)

        if n_real > 0:
            idx = torch.randint(0, real_x.size(0), (n_real,), device=device)
            x_real = real_x[idx]
            y_real = real_y[idx]
            logits = surrogate(x_real)
            loss_s = loss_s + ce(logits, y_real)

        if consistency_weight > 0 and consistency_noise > 0:
            ref_x = x_gen if n_gen > 0 else x_real
            noise = torch.randn_like(ref_x) * consistency_noise
            logits_pert = surrogate(ref_x + noise)
            loss_cons = F.kl_div(
                F.log_softmax(logits_pert, dim=1),
                F.softmax(surrogate(ref_x).detach(), dim=1),
                reduction="batchmean",
            )
            loss_s = loss_s + consistency_weight * loss_cons

        opt_s.zero_grad()
        loss_s.backward()
        opt_s.step()

        # Maximize disagreement to explore boundary regions.
        z = torch.randn(batch_size, z_dim, device=device)
        x = generator(z)
        if query_budget is None or queries < query_budget:
            y = oracle.predict(x)
            queries += batch_size
            logits = surrogate(x)
            loss_g = -ce(logits, y)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if step % log_every == 0:
            print(f"[Stage1] step={step} loss_s={loss_s.item():.4f} loss_g={loss_g.item():.4f} queries={queries}")

    return SurrogateBundle(surrogate=surrogate, generator=generator)


def train_surrogate_blackbox(
    oracle,
    feature_dim: int,
    n_classes: int,
    z_dim: int,
    gen_hidden,
    sur_hidden,
    steps: int,
    batch_size: int,
    lr_s: float,
    lr_g: float,
    device: torch.device,
    log_every: int = 200,
    query_budget: int | None = None,
    consistency_weight: float = 0.0,
    consistency_noise: float = 0.0,
    update_generator: bool = True,
    use_forward_diff: bool = True,
    n_G: int = 1,
    n_S: int = 1,
    fd_m: int = 3,
    fd_epsilon: float = 0.01,
    query_strategy: str = "random",
    query_pool: int = 1,
    query_mix_ratio: float = 0.5,
    real_x: torch.Tensor | None = None,
    query_real_ratio: float = 0.0,
    query_balance: bool = False,
    query_label_noise: float = 0.0,
    real_warmup_steps: int = 0,
    extraction_rounds: int = 1,
):
    start_time = time.perf_counter()
    generator = MLP(z_dim, gen_hidden, feature_dim).to(device)
    surrogate = MLP(feature_dim, sur_hidden, n_classes).to(device)

    opt_s = torch.optim.Adam(surrogate.parameters(), lr=lr_s)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    ce = nn.CrossEntropyLoss()

    queries = 0
    round_log: list[dict[str, float]] = []
    round_start_time = time.perf_counter()
    round_start_queries = 0
    loss_g = torch.tensor(0.0, device=device)
    real_pool = None
    real_pool_device = None
    if real_x is not None and query_real_ratio > 0.0:
        real_pool = real_x
        real_pool_device = real_x.device

    def _apply_query_label_noise(y: torch.Tensor) -> torch.Tensor:
        if query_label_noise <= 0.0 or y.numel() == 0 or n_classes <= 1:
            return y
        noise_mask = torch.rand(y.shape[0], device=y.device) < float(query_label_noise)
        if not torch.any(noise_mask):
            return y
        noisy = y.clone()
        rand_offset = torch.randint(1, n_classes, (int(noise_mask.sum().item()),), device=y.device)
        noisy[noise_mask] = (noisy[noise_mask] + rand_offset) % n_classes
        return noisy

    def forward_difference_step(batch_size: int) -> tuple[torch.Tensor, float, int]:
        z = torch.randn(batch_size, z_dim, device=device, requires_grad=True)
        x_base = generator(z)
        with torch.no_grad():
            y_base = oracle.predict(x_base)
            base_loss = ce(surrogate(x_base), y_base)

        grad_approx = torch.zeros_like(z)
        for _ in range(fd_m):
            u = torch.randn_like(z, device=device)
            u_norm = u / (torch.norm(u, dim=1, keepdim=True) + 1.0e-8)
            z_perturbed = z + fd_epsilon * u_norm
            x_perturbed = generator(z_perturbed)
            with torch.no_grad():
                y_pert = oracle.predict(x_perturbed)
                pert_loss = ce(surrogate(x_perturbed), y_pert)
            directional = (pert_loss - base_loss) / fd_epsilon
            grad_approx = grad_approx + directional.view(-1, 1) * u_norm

        grad_approx = grad_approx / float(fd_m)
        return z, grad_approx, float(base_loss.item()), (fd_m + 1) * batch_size

    def _select_queries(x_pool: torch.Tensor, k: int) -> torch.Tensor:
        if k >= x_pool.size(0):
            return torch.arange(x_pool.size(0), device=x_pool.device)
        strat = str(query_strategy).lower()
        if strat in ("random", "", "none"):
            return torch.randperm(x_pool.size(0), device=x_pool.device)[:k]
        with torch.no_grad():
            logits = surrogate(x_pool)
            probs = F.softmax(logits, dim=1)
            if strat == "entropy":
                score = -torch.sum(probs * torch.log(probs + 1.0e-8), dim=1)
                return torch.topk(score, k).indices
            if strat == "margin":
                top2 = torch.topk(probs, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                return torch.topk(-margin, k).indices
            if strat == "mix":
                k_unc = int(round(k * query_mix_ratio))
                k_unc = max(1, min(k, k_unc))
                score = -torch.sum(probs * torch.log(probs + 1.0e-8), dim=1)
                top_unc = torch.topk(score, k_unc).indices
                remaining = torch.tensor(
                    [i for i in range(x_pool.size(0)) if i not in set(top_unc.tolist())],
                    device=x_pool.device,
                )
                if remaining.numel() == 0:
                    return top_unc
                k_rand = k - k_unc
                if k_rand <= 0:
                    return top_unc
                rand_idx = remaining[torch.randperm(remaining.numel(), device=x_pool.device)[:k_rand]]
                return torch.cat([top_unc, rand_idx], dim=0)
        return torch.randperm(x_pool.size(0), device=x_pool.device)[:k]

    def _balanced_select(x_pool: torch.Tensor, y_pool: torch.Tensor, k: int) -> torch.Tensor:
        classes = torch.unique(y_pool).tolist()
        if len(classes) <= 1:
            return _select_queries(x_pool, k)
        per = max(1, k // len(classes))
        chosen = []
        for c in classes:
            idx_c = (y_pool == c).nonzero(as_tuple=False).view(-1)
            if idx_c.numel() == 0:
                continue
            take = min(per, idx_c.numel())
            idx_pool = idx_c[_select_queries(x_pool[idx_c], take)]
            chosen.append(idx_pool)
        if not chosen:
            return _select_queries(x_pool, k)
        chosen_idx = torch.cat(chosen, dim=0)
        if chosen_idx.numel() >= k:
            return chosen_idx[:k]
        remaining = k - chosen_idx.numel()
        mask = torch.ones(x_pool.size(0), dtype=torch.bool, device=x_pool.device)
        mask[chosen_idx] = False
        leftover = torch.nonzero(mask, as_tuple=False).view(-1)
        if leftover.numel() > 0:
            extra = leftover[_select_queries(x_pool[leftover], min(remaining, leftover.numel()))]
            chosen_idx = torch.cat([chosen_idx, extra], dim=0)
        return chosen_idx

    if real_pool is not None and real_warmup_steps > 0:
        surrogate.train()
        for step in range(1, real_warmup_steps + 1):
            warm_batch = batch_size
            if query_budget is not None:
                remaining = query_budget - queries
                warm_batch = min(batch_size, remaining)
                if warm_batch <= 0:
                    break
            idx = torch.randint(0, real_pool.size(0), (warm_batch,), device=real_pool_device)
            x_batch = real_pool[idx]
            if real_pool_device != device:
                x_batch = x_batch.to(device)
            with torch.no_grad():
                y_batch = oracle.predict(x_batch)
                y_batch = _apply_query_label_noise(y_batch)
            queries += int(x_batch.size(0))
            logits = surrogate(x_batch)
            loss_warm = ce(logits, y_batch)
            opt_s.zero_grad()
            loss_warm.backward()
            opt_s.step()
            if step % max(1, log_every) == 0:
                print(f"[Stage1] warmup step={step} loss_s={loss_warm.item():.4f} queries={queries}")

    rounds = max(1, int(extraction_rounds))
    steps_per_round = max(1, int((steps + rounds - 1) // rounds))
    current_round = 1
    for step in range(1, steps + 1):
        # Update generator with query-driven zeroth-order gradient estimation.
        if update_generator and use_forward_diff:
            generator.train()
            surrogate.eval()
            for _ in range(n_G):
                if query_budget is not None and queries >= query_budget:
                    break
                fd_batch = batch_size
                if query_budget is not None:
                    remaining = query_budget - queries
                    fd_cost = fd_m + 1
                    fd_batch = min(batch_size, remaining // fd_cost)
                    if fd_batch <= 0:
                        break
                z, grad_approx, base_loss, q = forward_difference_step(fd_batch)
                queries += q
                opt_g.zero_grad()
                z.backward(grad_approx)
                opt_g.step()
                loss_g = torch.tensor(base_loss, device=device)

        # Update student with hard-label queries.
        surrogate.train()
        generator.eval()
        loss_s = torch.tensor(0.0, device=device)
        for _ in range(n_S):
            if query_budget is not None and queries >= query_budget:
                break
            k = batch_size
            if query_budget is not None:
                k = min(k, query_budget - queries)
                if k <= 0:
                    break
            n_real = 0
            if real_pool is not None and query_real_ratio > 0.0:
                n_real = int(round(k * query_real_ratio))
                n_real = min(n_real, k)
            n_gen = k - n_real
            pool_mult = max(1, int(query_pool))
            x_gen = None
            y_gen = None
            if n_gen > 0:
                pool_size = max(n_gen, pool_mult * n_gen)
                if query_budget is not None:
                    remaining = query_budget - queries
                    pool_size = min(pool_size, remaining)
                if pool_size <= 0:
                    break
                z_pool = torch.randn(pool_size, z_dim, device=device)
                with torch.no_grad():
                    x_pool = generator(z_pool)
                with torch.no_grad():
                    y_pool = oracle.predict(x_pool)
                    y_pool = _apply_query_label_noise(y_pool)
                queries += int(x_pool.size(0))
                if query_balance:
                    idx = _balanced_select(x_pool, y_pool, n_gen)
                else:
                    idx = _select_queries(x_pool, n_gen)
                x_gen = x_pool[idx]
                y_gen = y_pool[idx]
            x_real = None
            y_real = None
            if n_real > 0 and real_pool is not None:
                pool_size = max(n_real, pool_mult * n_real)
                if query_budget is not None:
                    remaining = query_budget - queries
                    pool_size = min(pool_size, remaining)
                if pool_size <= 0:
                    break
                idx_pool = torch.randint(0, real_pool.size(0), (pool_size,), device=real_pool_device)
                x_pool = real_pool[idx_pool]
                if real_pool_device != device:
                    x_pool = x_pool.to(device)
                with torch.no_grad():
                    y_pool = oracle.predict(x_pool)
                    y_pool = _apply_query_label_noise(y_pool)
                queries += int(x_pool.size(0))
                if query_balance:
                    idx = _balanced_select(x_pool, y_pool, n_real)
                else:
                    idx = _select_queries(x_pool, n_real)
                x_real = x_pool[idx]
                y_real = y_pool[idx]
            if x_gen is not None and x_real is not None:
                x_gen = torch.cat([x_gen, x_real], dim=0)
                y_gen = torch.cat([y_gen, y_real], dim=0)
            elif x_gen is None and x_real is not None:
                x_gen = x_real
                y_gen = y_real
            logits = surrogate(x_gen)
            loss_step = ce(logits, y_gen)

            if consistency_weight > 0 and consistency_noise > 0:
                noise = torch.randn_like(x_gen) * consistency_noise
                logits_pert = surrogate(x_gen + noise)
                loss_cons = F.kl_div(
                    F.log_softmax(logits_pert, dim=1),
                    F.softmax(surrogate(x_gen).detach(), dim=1),
                    reduction="batchmean",
                )
                loss_step = loss_step + consistency_weight * loss_cons

            opt_s.zero_grad()
            loss_step.backward()
            opt_s.step()
            loss_s = loss_s + loss_step.detach()

        if step % log_every == 0:
            print(f"[Stage1] step={step} loss_s={loss_s.item():.4f} loss_g={loss_g.item():.4f} queries={queries}")

        reached_round_end = step >= min(steps, current_round * steps_per_round)
        if reached_round_end or (query_budget is not None and queries >= query_budget):
            now = time.perf_counter()
            round_log.append(
                {
                    "round": float(current_round),
                    "end_step": float(step),
                    "query_count_total": float(queries),
                    "query_count_delta": float(queries - round_start_queries),
                    "runtime_sec_delta": float(now - round_start_time),
                    "loss_s": float(loss_s.item()),
                    "loss_g": float(loss_g.item()),
                }
            )
            current_round += 1
            round_start_time = now
            round_start_queries = queries

        if query_budget is not None and queries >= query_budget:
            break

    return SurrogateBundle(
        surrogate=surrogate,
        generator=generator,
        query_count=int(queries),
        runtime_sec=float(time.perf_counter() - start_time),
        round_log=round_log,
    )
