import numpy as np
import torch
import torch.nn.functional as F
from skrl.resources.schedulers.torch import KLAdaptiveLR

from learn.resnet import ResNet


class Agent:
    def __init__(self, args):
        self.save_path = args.save_path
        self.model_name = f"resnet_{args.width}_{args.height}_{args.n_in_row}.pkl"
        self.args = args
        self.net = ResNet(args).cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = KLAdaptiveLR(self.optimizer, min_lr=args.min_lr, max_lr=args.max_lr, kl_threshold=args.kl_threshold)
        if args.load_model:
            self.net.load_state_dict(torch.load(self.save_path / self.model_name))

    @torch.no_grad()
    def predict(self, board):
        state = np.ascontiguousarray(board.get_full_state())
        state = torch.from_numpy(state).cuda().unsqueeze(0).float()
        probs, value = self.net(state)  # inference
        probs = np.exp(probs.detach().cpu().numpy().flatten())
        value = value.detach().cpu().numpy().item()

        action_probs = zip(board.availables, probs[board.availables])
        return action_probs, value

    @torch.no_grad()
    def predict_batch(self, states):
        states = torch.from_numpy(states).cuda().float()
        probs, value = self.net(states)
        probs = np.exp(probs.detach().cpu().numpy())
        value = value.detach().cpu().numpy()
        return probs, value

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save_model(self):
        torch.save(self.net.state_dict(), self.save_path / self.model_name)
        print("[INFO] model saved")

    def train_step(self, states, mcts_probs, winners_z):
        states = torch.from_numpy(states).cuda().float()
        mcts_probs = torch.from_numpy(mcts_probs).cuda().float()
        winners_z = torch.from_numpy(winners_z).cuda().float()
        self.optimizer.zero_grad()
        probs, value = self.net(states)
        loss = F.mse_loss(value.view(-1), winners_z) - torch.mean(torch.sum(probs * mcts_probs, dim=1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
