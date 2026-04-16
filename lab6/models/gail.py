import numpy as np
import torch
import logging
import os

from torch.nn import Module

from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import (
    get_flat_grads,
    get_flat_params,
    set_params,
    conjugate_gradient,
    rescale_and_linesearch,
)

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


def setup_logger(log_dir="logs", log_file="gail_train.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("GAIL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)

    return logger


class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        # ============================================================
        # TODO (Students):
        # Initialize:
        #   - self.pi  (PolicyNetwork)
        #   - self.v   (ValueNetwork)
        #   - self.d   (Discriminator)
        #
        # Use appropriate input/output dimensions.
        # ============================================================
        self.pi = None
        self.v = None
        self.d = None

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        distb = self.pi(state)

        # ============================================================
        # TODO (Students):
        # Sample action from distb and process it:
        #   1. distb.sample()
        #   2. detach, move to cpu, convert to numpy
        #   3. squeeze batch dimension
        #   4. clip to [-1, 1]
        # ============================================================
        action = None

        return action

    def _collect_trajectories(self, env, policy_fn, num_steps, horizon, render=False):
        all_obs = []
        all_acts = []
        all_rwds_per_ep = []

        steps = 0
        while steps < num_steps:
            ep_rwds = []

            t = 0
            done = False
            ob, info = env.reset()

            while not done and steps < num_steps:
                act = policy_fn(ob)
                act = np.asarray(act, dtype=np.float32).reshape(-1)

                all_obs.append(ob)
                all_acts.append(act)

                if render:
                    env.render()

                ob, rwd, terminated, truncated, info = env.step(act)
                done = terminated or truncated

                ep_rwds.append(rwd)
                t += 1
                steps += 1

                if horizon is not None and t >= horizon:
                    done = True
                    break

            if len(ep_rwds) > 0:
                all_rwds_per_ep.append(np.sum(ep_rwds))

        obs_tensor = FloatTensor(np.array(all_obs))
        acts_tensor = FloatTensor(np.array(all_acts))
        return obs_tensor, acts_tensor, all_rwds_per_ep

    def _sample_expert_batch(self, expert_obs, expert_acts, batch_size):
        n = expert_obs.shape[0]
        idx = np.random.randint(0, n, size=batch_size)
        return expert_obs[idx], expert_acts[idx]

    @torch.no_grad()
    def compute_test_metrics(
        self,
        test_env,
        expert_obs,
        expert_acts,
        num_steps,
        horizon,
    ):
        # sample expert test batch from fixed expert dataset
        exp_obs, exp_acts = self._sample_expert_batch(
            expert_obs, expert_acts, num_steps
        )

        # collect generator test trajectories
        gen_obs, gen_acts, gen_rwds = self._collect_trajectories(
            test_env, self.act, num_steps, horizon
        )

        self.pi.eval()
        distb = self.pi(exp_obs)
        imitation_logp = distb.log_prob(exp_acts)
        if imitation_logp.ndim > 1:
            imitation_logp = imitation_logp.sum(-1)
        imitation_loss = (-imitation_logp).mean().item()

        self.d.eval()
        exp_scores = self.d.get_logits(exp_obs, exp_acts)
        gen_scores = self.d.get_logits(gen_obs, gen_acts)

        # keep same labeling convention as your original code
        disc_loss_expert = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_scores, torch.zeros_like(exp_scores)
        ).item()
        disc_loss_gen = torch.nn.functional.binary_cross_entropy_with_logits(
            gen_scores, torch.ones_like(gen_scores)
        ).item()
        disc_loss = disc_loss_expert + disc_loss_gen

        gen_rwd_mean = np.mean(gen_rwds) if gen_rwds else 0.0

        return {
            "imitation_loss": imitation_loss,
            "disc_loss": disc_loss,
            "disc_loss_expert": disc_loss_expert,
            "disc_loss_generated": disc_loss_gen,
            "gen_test_reward_mean": gen_rwd_mean,
        }

    def train(
        self,
        env,
        expert_obs,
        expert_acts,
        render=False,
        test_env=None,
    ):
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]
        test_steps = self.train_config.get("num_test_steps", num_steps_per_iter)
        log_file = self.train_config["log_file"]

        self.logger = setup_logger(log_file=log_file)

        opt_d = torch.optim.Adam(self.d.parameters())

        # fixed expert dataset
        expert_obs = FloatTensor(np.array(expert_obs))
        expert_acts = FloatTensor(np.array(expert_acts))
        expert_dataset_size = expert_obs.shape[0]

        self.logger.info(
            "Training started | num_expert_samples=%d",
            expert_dataset_size,
        )
        self.logger.info(
            "iter | train_reward | test_imitation_loss | test_disc_loss "
            "| test_disc_loss_expert | test_disc_loss_generated "
            "| test_gen_reward"
        )

        rwd_iter_means = []

        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False
                ob, info = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)
                    act = np.asarray(act, dtype=np.float32).reshape(-1)

                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    if render:
                        env.render()

                    ob, rwd, terminated, truncated, info = env.step(act)
                    done = terminated or truncated

                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)

                    t += 1
                    steps += 1

                    if horizon is not None and t >= horizon:
                        done = True
                        break

                if len(ep_rwds) == 0:
                    continue

                rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[j:]) for j in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms
                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.0]]))
                ).detach()

                ep_deltas = ep_costs.unsqueeze(-1) + gae_gamma * next_vals - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[: t - j].unsqueeze(-1) * ep_deltas[j:]).sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)
                gms.append(ep_gms)

            rwd_iter_mean = np.mean(rwd_iter) if len(rwd_iter) > 0 else 0.0
            rwd_iter_means.append(rwd_iter_mean)
            print(f"Iterations: {i + 1},   Reward Mean: {rwd_iter_mean}")

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # -------------------------
            # Update discriminator
            # -------------------------
            self.d.train()

            batch_exp_obs, batch_exp_acts = self._sample_expert_batch(
                expert_obs, expert_acts, obs.shape[0]
            )

            exp_scores = self.d.get_logits(batch_exp_obs, batch_exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            # ============================================================
            # TODO (Students):
            # Implement GAIL discriminator loss:
            #
            # Expert → label 0
            # Generated → label 1
            #
            # Use BCE with logits
            # loss = (
            #     torch.nn.functional.binary_cross_entropy_with_logits(
            #         TODO, torch.zeros_like(TODO)
            #     )
            #     + torch.nn.functional.binary_cross_entropy_with_logits(
            #         TODO, torch.ones_like(TODO)
            #     )
            # )
            # ============================================================
            loss = None
            
            loss.backward()
            opt_d.step()

            # -------------------------
            # Update value function
            # -------------------------
            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / (torch.dot(s, Hs) + 1e-8))

            new_params = old_params + alpha * s
            set_params(self.v, new_params)

            # -------------------------
            # Update policy
            # -------------------------
            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def _log_prob(distb, x):
                lp = distb.log_prob(x)
                if lp.ndim > 1:
                    lp = lp.sum(-1)
                return lp

            def L():
                distb = self.pi(obs)
                return (
                    advs.squeeze(-1)
                    * torch.exp(
                        _log_prob(distb, acts) - _log_prob(old_distb, acts).detach()
                    )
                ).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs
                    return (old_p * (torch.log(old_p) - torch.log(p))).sum(-1).mean()
                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return 0.5 * (
                        (old_cov / cov).sum(-1)
                        + (((old_mean - mean) ** 2) / cov).sum(-1)
                        - self.action_dim
                        + torch.log(cov).sum(-1)
                        - torch.log(old_cov).sum(-1)
                    ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi,
                ).detach()
                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()
            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * _log_prob(self.pi(obs), acts)).mean()
            grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

            # -------------------------
            # Test-set evaluation
            # -------------------------
            eval_env = test_env if test_env is not None else env
            metrics = self.compute_test_metrics(
                eval_env,
                expert_obs,
                expert_acts,
                test_steps,
                horizon,
            )
            self.logger.info(
                "%d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f",
                i + 1,
                rwd_iter_mean,
                metrics["imitation_loss"],
                metrics["disc_loss"],
                metrics["disc_loss_expert"],
                metrics["disc_loss_generated"],
                metrics["gen_test_reward_mean"],
            )

        return rwd_iter_means