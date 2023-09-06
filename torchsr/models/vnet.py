
import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, nn.Parameter(tmp))
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, nn.Parameter(tmp))
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class VNet(MetaModule):
    def __init__(self, n_feats):
        super(VNet, self).__init__()
        
        n_colors = 3
        
        # 128x3x96x96 input loss
        # 1x1 conv 확장
        # 128x64x96x96
        # relu
        # 1x1 conv
        # weight bias initialize 0으로 초기화

        self.conv1 = nn.Conv2d(n_colors, n_feats, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(n_feats, 1, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()
        # self.linear1 = nn.Linear(1, n_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(n_feats, 1)
        # self.linear3 = nn.Linear(hidden2, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = self.linear1(x)
        # x = self.relu1(x)
        # out = self.linear2(x)

        # x = self.conv1(x)
        # x = self.relu1(x)
        # out = self.conv2(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.global_avgpool(x)
        x = self.conv3(x)
        out = self.relu1(x)
        
        return torch.sigmoid(out)
    
    # def params(self):
    #     for name, param in self.named_params(self):
    #         yield param

    # def named_leaves(self):
    #     return []

    # def named_submodules(self):
    #     return []

    # def named_params(self, curr_module=None, memo=None, prefix=''):
    #     if memo is None:
    #         memo = set()

    #     if hasattr(curr_module, 'named_leaves'):
    #         for name, p in curr_module.named_leaves():
    #             if p is not None and p not in memo:
    #                 memo.add(p)
    #                 yield prefix + ('.' if prefix else '') + name, p
    #     else:
    #         for name, p in curr_module._parameters.items():
    #             if p is not None and p not in memo:
    #                 memo.add(p)
    #                 yield prefix + ('.' if prefix else '') + name, p

    #     for mname, module in curr_module.named_children():
    #         submodule_prefix = prefix + ('.' if prefix else '') + mname
    #         for name, p in self.named_params(module, memo, submodule_prefix):
    #             yield name, p

    # def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
    #     if source_params is not None:
    #         for tgt, src in zip(self.named_params(self), source_params):
    #             name_t, param_t = tgt
    #             # name_s, param_s = src
    #             # grad = param_s.grad
    #             # name_s, param_s = src
    #             grad = src
    #             if first_order:
    #                 grad = to_var(grad.detach().data)
    #             tmp = param_t - lr_inner * grad
    #             breakpoint()
    #             self.set_param(self, name_t, tmp)
    #     else:

    #         for name, param in self.named_params(self):
    #             if not detach:
    #                 grad = param.grad
    #                 if first_order:
    #                     grad = to_var(grad.detach().data)
    #                 tmp = param - lr_inner * grad
    #                 self.set_param(self, name, tmp)
    #             else:
    #                 param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
    #                 self.set_param(self, name, param)

    # def set_param(self, curr_mod, name, param):
    #     if '.' in name:
    #         n = name.split('.')
    #         module_name = n[0]
    #         rest = '.'.join(n[1:])
    #         for name, mod in curr_mod.named_children():
    #             if module_name == name:
    #                 self.set_param(mod, rest, param)
    #                 break
    #     else:
    #         setattr(curr_mod, name, param)

    # def detach_params(self):
    #     for name, param in self.named_params(self):
    #         self.set_param(self, name, param.detach())

    # def copy(self, other, same_var=False):
    #     for name, param in other.named_params():
    #         if not same_var:
    #             param = to_var(param.data.clone(), requires_grad=True)
    #         self.set_param(name, param)


class VNet2(nn.Module):
    def __init__(self, n_feats):
        super(VNet2, self).__init__()
        
        n_colors = 3
        # self.conv1 = nn.Conv2d(n_colors, n_feats, 3, padding=1, bias=False)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(n_feats, n_feats*2, 3, padding=1, bias=False)
        # self.conv3 = nn.Conv2d(n_feats*2, n_feats, 3, padding=1, bias=False)
        # self.conv4 = nn.Conv2d(n_feats, n_colors, 3, padding=1, bias=False)

        input = 1
        output = 1
        self.conv1 = nn.Conv2d(n_colors, n_feats, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_feats, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_mean = nn.Conv2d(n_feats, 1) 
        # self.conv_var = nn.Conv2d(n_feats, 1) 
        self.cls_emb = nn.Embedding(output, output)


        # self.linear1 = nn.Linear(input, n_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(n_feats, n_feats)
        # self.linear_mean = nn.Linear(n_feats, output) 
        # self.linear_var = nn.Linear(n_feats, output) 
        # self.cls_emb = nn.Embedding(output, output)

        self.init_weights()
        # self.linear3 = nn.Linear(hidden2, 1)

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        # torch.nn.init.xavier_normal_(self.linear1.weight)
        # self.linear1.bias.data.zero_()
        # torch.nn.init.xavier_normal_(self.linear2.weight)
        # self.linear2.bias.data.zero_()
        # torch.nn.init.xavier_normal_(self.linear_mean.weight)
        # self.linear_mean.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.linear_mean(h2)
        log_var = self.linear_var(h2)
        return mean, log_var

    def forward(self, feat, target, sample_num):
        target = self.cls_emb(target)

        x = torch.cat([feat, target], dim=-1)

        mean, log_var = self.encode(x)  # or 100
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std.unsqueeze(0).repeat(sample_num,1,1))


        return torch.sigmoid(mean + std*eps)
    
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)