import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...environment import BaseEnvironment

class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(4, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 8, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {'policy': p, 'value': v}

class Environment(BaseEnvironment):
    ACTION = ['MWEST', 'MEAST', 'MNORTH', 'MSOUTH', 'NOTHING', 'PWEST', 'PEAST', 'PNORTH', 'PSOUTH']
    NUM_AGENTS = 2
    COLORS = [[0.5, 0.5, 0], [0.5, 0, 0.5], [0.25, 0.25, 0.], [0.25, 0., 0.25]]
    MAX_STEPS = 100
    def __init__(self, args={}):
        super().__init__()
        # action space is (move left, move right, move up, move down, stay)
        # and (place obstacle left, right, up, down, don't place obstacle)
        # therefore [5,5]
        self.multidiscrete = False
        self.nvec = [5,5]
        self.reset()

    def reset(self, args={}):
        self.board = np.zeros((10,10))
        self._players = {}
        self.steps = 0
        # select initial pos uniformly randomly
        # must be careful cause selection of position
        # for the secondd player depends on position of first
        # because two players cannot be on the same tile
        # at the same time
        pos = np.array([np.tile(np.arange(10), 10), np.repeat(np.arange(10), 10)]).T
        idxs = np.arange(10*10).tolist()
        for p in self.players():
            choice = idxs.pop(np.random.choice(idxs))
            self._players[p] = {"pos": pos[choice]}
            self.board[pos[choice][0], pos[choice][1]] = p+1
            
        obs = self.board
        self.update((obs, {}), True)

    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs.copy())
        self.last_actions = last_actions
    
    def target_pos(self, cur_pos, action):
        # left
        if action == 0:
            return [(cur_pos[0]-1)%10, cur_pos[1]]
        # right
        elif action == 1:
            return [(cur_pos[0]+1)%10, cur_pos[1]]
        # up
        elif action == 2:
            return [cur_pos[0], (cur_pos[1]+1)%10]
        #down
        elif action == 3:
            return [cur_pos[0], (cur_pos[1]-1)%10]
        # stay
        elif action == 4:
            return cur_pos
    
    def step(self, actions):
        if self.multidiscrete:
            return self.multidiscrete_step(actions)
        else:
            return self.discrete_step(actions)
    
    def discrete_step(self, actions):
        dests = {}
        # select target position for the action
        for p in self.players():
            action = actions.get(p, None) or 0
            cur_pos = self._players[p]["pos"]
            # move action
            if action <= 4:
                dests[p] = {
                    "move_dest": self.target_pos(cur_pos, action),
                    "place_dest": None
                }
            # place action
            else:
                dests[p] = {
                    "move_dest": None,
                    "place_dest": self.target_pos(cur_pos, action-5)
                }
            
        for p1 in self.players():
            if not dests[p1]["move_dest"] is None:
                # can move if nothing at target position
                valid_move = not self.board[dests[p1]["move_dest"][0], dests[p1]["move_dest"][1]]

                if valid_move:
                    for p2 in self.players():
                        if p1 == p2:
                            continue
                        # cannot move if another player wants to move to the same place
                        if not dests[p2]["move_dest"] is None:
                            if dests[p1]["move_dest"][0] == dests[p2]["move_dest"][0] and \
                                dests[p1]["move_dest"][1] == dests[p2]["move_dest"][1]:
                                valid_move = False

                if valid_move:
                    cur_pos = self._players[p1]["pos"]
                    self.board[cur_pos[0], cur_pos[1]] = 0
                    self.board[dests[p1]["move_dest"][0], dests[p1]["move_dest"][1]] = p1+1
                    self._players[p1]["pos"] = dests[p1]["move_dest"]
            else:
                # can place if nothing at target position
                valid_place = not self.board[dests[p1]["place_dest"][0], dests[p1]["place_dest"][1]]
                
                if valid_place:
                    for p2 in self.players():
                        if p1 == p2:
                            continue

                        # cannot place if another player wants to place a tile to target dir
                        if not dests[p2]["place_dest"] is None:
                            if (dests[p1]["place_dest"][0] == dests[p2]["place_dest"][0] and \
                                dests[p1]["place_dest"][1] == dests[p2]["place_dest"][1]):
                                valid_place = False

                if valid_place:
                    self.board[dests[p1]["place_dest"][0], dests[p1]["place_dest"][1]] = self.NUM_AGENTS+2+p1
        
        self.steps += 1
        obs = self.board
        self.update((obs, actions), False)
    
    def multidiscrete_step(self, actions):
        # state transition
        dests = {}
        # select target position for both actions
        for p in self.players():
            action = actions.get(p, None) or 0
            move_action = action[0]
            place_action = action[1]
            cur_pos = self._players[p]["pos"]
            dests[p] = {
                "move_dest": self.target_pos(cur_pos, move_action),
                "place_dest": self.target_pos(cur_pos, place_action) if place_action != 4 else None
            }

        for p1 in self.players():
            # can move if don't place at target position and if nothing at target position
            valid_move = not self.board[dests[p1]["move_dest"][0], dests[p1]["move_dest"][1]] and \
                        (dests[p1]["place_dest"] is None or \
                            (dests[p1]["move_dest"][0] != dests[p1]["place_dest"][0] or \
                             dests[p1]["move_dest"][1] != dests[p1]["place_dest"][1]))
                        
            if valid_move:
                for p2 in self.players():
                    if p1 == p2:
                        continue
                    # cannot move if another player wants to move to the same place
                    if dests[p1]["move_dest"][0] == dests[p2]["move_dest"][0] and \
                        dests[p1]["move_dest"][1] == dests[p2]["move_dest"][1]:
                        valid_move = False
                        
            if valid_move:
                cur_pos = self._players[p1]["pos"]
                self.board[cur_pos[0], cur_pos[1]] = 0
                self.board[dests[p1]["move_dest"][0], dests[p1]["move_dest"][1]] = p1+1
                self._players[p1]["pos"] = dests[p1]["move_dest"]
            
            # can place if nothing at target position
            valid_place = dests[p1]["place_dest"] is None or \
                            not self.board[dests[p1]["place_dest"][0], dests[p1]["place_dest"][1]]
            if valid_place and not dests[p1]["place_dest"] is None:
                for p2 in self.players():
                    if p1 == p2:
                        continue
                    
                    # cannot place if another player wants to place a tile to target dir
                    if not dests[p2]["place_dest"] is None:
                        if (dests[p1]["place_dest"][0] == dests[p2]["place_dest"][0] and \
                            dests[p1]["place_dest"][1] == dests[p2]["place_dest"][1]):
                            valid_place = False
            
            if valid_place and not dests[p1]["place_dest"] is None:
                self.board[dests[p1]["place_dest"][0], dests[p1]["place_dest"][1]] = self.NUM_AGENTS+2+p1
        self.steps += 1
        obs = self.board
        self.update((obs, actions), False)

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        return self.players()

    def terminal(self):
        # check whether terminal state or not
        if (self.board == 0).any() and self.steps <= self.MAX_STEPS:
            return False
        
        return True

    def outcome(self):
        # return terminal outcomes
        rewards = {p: (self.obs_list[-1]==self.NUM_AGENTS+2+p).sum() for p in self.players()}
        outcomes = {p: 0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes
            
    
    def render(self):
        # then use :
        # ffmpeg -r 1 -f image2 -s 1920x1080 -i img_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
        # to get the resulting video
        for j,obs in enumerate(self.obs_list):
            img = np.zeros((10,10,3))
            for p in self.players():
                t = np.argwhere(obs==p+1)
                img[t[:,0], t[:,1], :] = self.COLORS[p]
                t = np.argwhere(obs==self.NUM_AGENTS+2+p)
                img[t[:,0], t[:,1], :] = self.COLORS[self.NUM_AGENTS+p]
            
            Image.fromarray((img*255).astype(np.uint8)).resize([300, 300], resample=Image.NEAREST).save(f"img_{j}.png")
                
    def legal_actions(self, player):
        # return legal action list
        cur_pos = self._players[player]["pos"]
        la = np.arange(9).tolist()
        for i in range(9):
            if i <= 3:
                target_pos = self.target_pos(cur_pos, i)
                if self.board[target_pos[0], target_pos[1]]:
                    la.remove(i)
            elif i>4:
                target_pos = self.target_pos(cur_pos, i-5)
                if self.board[target_pos[0], target_pos[1]]:
                    la.remove(i)
            
        return la

    def players(self):
        return list(range(self.NUM_AGENTS))

    def net(self):
        return GeeseNet()

    def observation(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 2, 10, 10), dtype=np.float32)
        obs = self.obs_list[-1]

        for p in self.players():
            b[0 + (p - player) % self.NUM_AGENTS] = (obs == p+1).astype(np.uint8)
            b[self.NUM_AGENTS + (p - player) % self.NUM_AGENTS] = (obs == self.NUM_AGENTS + 2 + p).astype(np.uint8)

        return b

if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = {p: e.legal_actions(p) for p in e.turns()}
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e)
        print(e.outcome())
