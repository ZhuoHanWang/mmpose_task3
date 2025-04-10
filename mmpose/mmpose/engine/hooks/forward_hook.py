from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Union

import torch
from torch import nn  # 补充对 torch.nn 的导入

@HOOKS.register_module()
class ForwardHook(Hook):
    """A hook to monitor the forward outputs of specific modules."""

    def __init__(self, module: str):
        self.module_name = module

    def before_run(self, runner):
        """Attach the forward hook to the specified module."""
        # 打印模型结构以调试模块路径
        print(f"Attempting to attach hook to module: {self.module_name}")
        print("Available modules in the model:")
        for name, module in runner.model.named_modules():
            print(name)

        # 查找模块并绑定钩子
        module = self._find_module(runner.model, self.module_name)
        if module is not None:
            module.register_forward_hook(self._print_output)
        else:
            raise ValueError(f"Module '{self.module_name}' not found in the model.")

    def _find_module(self, model, module_name):
        """Find a module by its name, supporting nested structures and lists."""
        parts = module_name.split('.')
        current_module = model
        for part in parts:
            # 支持列表或 ModuleList 的索引访问
            if isinstance(current_module, (list, nn.ModuleList)):
                try:
                    part = int(part)  # 将索引字符串转换为整数
                    current_module = current_module[part]
                except (ValueError, IndexError):
                    return None
            elif hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                return None
        return current_module

    def _print_output(self, module, input, output):
        print(f"{module.__class__.__name__} output shape: {output.shape}")
