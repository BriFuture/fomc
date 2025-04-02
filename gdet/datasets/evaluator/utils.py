## bsf.c 仿照 jax.tree_map 将 字典转为 np.array
from typing import Sequence, Mapping, AbstractSet
import numpy as np
import torch

class TreeNode():
    def __init__(self, item: "dict", name="", parent: "TreeNode"=None):
        self.item = item
        self.name = name
        self._parent = parent
        self._childeren: "list[TreeNode]" = []
        if parent is not None:
            parent._childeren.append(self)

    def dtype(self):
        return type(self.item)
    
    def is_leave(self):
        return type(self.item) in (torch.Tensor, int, float, np.ndarray)
    
    def is_container(self):
        return type(self.item) in (dict, list, tuple)
    
    def parent(self):
        if self._parent:
            return self._parent
        return None
    def parent_name(self):
        if self._parent:
            return self._parent.name
        return "None"
    def __repr__(self):
        return f'<Node: {self.name} children: {len(self._childeren)}>'
    
    def attach_to_parent(self):
        if self._parent is None:
            return
        dt = self._parent.dtype()
        pitem = self._parent.item
        if dt is dict:
            pitem[self.name] = self.item
        elif dt is list:
            pitem: "list"
            pitem.append(self.item)
        else:
            raise ValueError("Unsupported Parent Item Type")
def detach_tensor(t: "torch.Tensor"):
    dtype = type(t)
    if dtype is torch.Tensor:
        return t.detach().cpu().numpy()
    elif dtype in (float, int):
        return np.array(t)

    return t

def tree_map(func, embed_dict):
    pass
    root_node = TreeNode(embed_dict, "root")

    node_stack = [root_node]
    while len(node_stack):
        curr_node = node_stack.pop()
        if curr_node.is_container():
            if curr_node.dtype() is dict:
                for k, v in curr_node.item.items():
                    next_node = TreeNode(v, k, parent=curr_node)
                    node_stack.append(next_node)
            elif curr_node.dtype() in (list, tuple):
                for k, v in enumerate(curr_node.item):
                    next_node = TreeNode(v, f"{curr_node.name}/{k}", parent=curr_node)
                    node_stack.append(next_node)
        else:
            # print(curr_node.name, ", All Zero: ", curr_node.dtype(), curr_node.parent_name())
            ## replace
            pass
    # print(root_node)

    new_dict = {}
    new_root_node = TreeNode(new_dict, "root")
    node_stack = [root_node]
    new_node_stack = [new_root_node]
    while len(node_stack):
        curr_node = node_stack.pop()
        new_curr_node = new_node_stack.pop()
        for cn in curr_node._childeren:
            cn : "TreeNode"
            cntype = cn.dtype()
            node_stack.append(cn)
            if type(cn.name) is str and "/" in cn.name:
                ridx = cn.name.rindex("/")
                new_node_name = cn.name[ridx+1:]
            else:
                new_node_name = cn.name
            if cntype is dict:
                new_node = TreeNode({}, name=new_node_name, parent=new_curr_node)
            elif cntype in (list, tuple):
                new_node = TreeNode([], name=new_node_name, parent=new_curr_node)
            else:
                new_node = TreeNode(func(cn.item), name=new_node_name, parent=new_curr_node)
            new_node.attach_to_parent()
            new_node_stack.append(new_node)
    ##
    # print(new_dict)
    return new_dict