# from addict import Dict as BaseConfigDict
import copy

class BaseConfigDict(dict):

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        isFrozen = (hasattr(self, '__frozen') and
                    object.__getattribute__(self, '__frozen'))
        if isFrozen and name not in super(BaseConfigDict, self).keys():
                raise KeyError(name)
        super(BaseConfigDict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise KeyError(name)
        return self.__class__(__parent=self, __key=name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                (not isinstance(self[k], dict)) or
                (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __or__(self, other):
        if not isinstance(other, (BaseConfigDict, dict)):
            return NotImplemented
        new = BaseConfigDict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, (BaseConfigDict, dict)):
            return NotImplemented
        new = BaseConfigDict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def freeze(self, shouldFreeze=True):
        object.__setattr__(self, '__frozen', shouldFreeze)
        for key, val in self.items():
            if isinstance(val, BaseConfigDict):
                val.freeze(shouldFreeze)

    def unfreeze(self):
        self.freeze(False)


from types import FunctionType, ModuleType

class ConfigDict(BaseConfigDict):
    """增加 immutable 功能, GROUP 功能"""
    IMMUTABLE = "__immutable__"
    GROUP = "__group__"
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, ConfigDict.IMMUTABLE, False)
        super().__init__(*args, **kwargs)
    
    def __missing__(self, name):
        raise KeyError(name)
    
    def __getattr__(self, name: "str"):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

        
    def is_immutable(self):
        return object.__getattribute__(self, ConfigDict.IMMUTABLE)
    
    def set_immutable(self, immutable):
        self.__setitem__(ConfigDict.IMMUTABLE, immutable)

    def freeze(self):
        """freeze 后，所有 item 都无法设置（原版是存在的key-value可以设置）"""
        self.set_immutable(True)

    def defrost(self):
        self.set_immutable(False)

    def __setitem__(self, name, value):
        if isinstance(value, (FunctionType, ModuleType)): #bsf.c 移除 函数，模块类型
            return
        if name == ConfigDict.IMMUTABLE:
            object.__setattr__(self, ConfigDict.IMMUTABLE, value)
            # bsf.c 子类 freeze
            for v in self.values():
                if isinstance(v, ConfigDict):
                    v[name] = value
        elif name == ConfigDict.GROUP:
            # self.set_group(value)
            pass
        else:
            is_mutable = (hasattr(self, ConfigDict.IMMUTABLE) and
                    object.__getattribute__(self, ConfigDict.IMMUTABLE))
            if is_mutable:
                raise ValueError("Config Is Frozen!")
            return super().__setitem__(name, value)
    
    def clone(self):
        return copy.deepcopy(self)