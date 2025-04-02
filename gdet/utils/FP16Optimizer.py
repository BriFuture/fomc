from torch.nn.utils import clip_grad
import copy
from collections import defaultdict
from itertools import chain

class OptimizerHook(object):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

class LossScaler:
    """Class that manages loss scaling in mixed precision training which
    supports both dynamic or static mode.

    The implementation refers to
    https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py.
    Indirectly, by supplying ``mode='dynamic'`` for dynamic loss scaling.
    It's important to understand how :class:`LossScaler` operates.
    Loss scaling is designed to combat the problem of underflowing
    gradients encountered at long times when training fp16 networks.
    Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.
    If overflowing gradients are encountered, :class:`FP16_Optimizer` then
    skips the update step for this particular iteration/minibatch,
    and :class:`LossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients
    detected,:class:`LossScaler` increases the loss scale once more.
    In this way :class:`LossScaler` attempts to "ride the edge" of always
    using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float): Initial loss scale value, default: 2**32.
        scale_factor (float): Factor used when adjusting the loss scale.
            Default: 2.
        mode (str): Loss scaling mode. 'dynamic' or 'static'
        scale_window (int): Number of consecutive iterations without an
            overflow to wait before increasing the loss scale. Default: 1000.
    """

    def __init__(self,
                 init_scale=2**32,
                 mode='dynamic',
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        assert mode in ('dynamic',
                        'static'), 'mode can only be dynamic or static'
        self.mode = mode
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def has_overflow(self, params):
        """Check if params contain overflow."""
        if self.mode != 'dynamic':
            return False
        for p in params:
            if p.grad is not None and LossScaler._has_inf_or_nan(p.grad.data):
                return True
        return False

    def _has_inf_or_nan(x):
        """Check if params contain NaN."""
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') \
                    or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        """update the current loss scale value when overflow happens."""
        if self.mode != 'dynamic':
            return
        if overflow:
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % \
                    self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float | str): Scale factor multiplied with loss. If
            'dynamic' is specified, then dynamic loss scaling will be used.
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        if loss_scale == 'dynamic':
            self.loss_scaler = LossScaler(mode='dynamic')
        elif isinstance(loss_scale, float):
            self.loss_scaler = LossScaler(init_scale=loss_scale, mode='static')
        else:
            raise ValueError('loss_scale must be of type float or str')

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training.

        1. Make a master copy of fp32 weights for optimization.
        2. Convert the main model from fp32 to fp16.
        """
        # keep a copy of fp32 weights
        old_groups = runner.optimizer.param_groups
        runner.optimizer.param_groups = copy.deepcopy(
            runner.optimizer.param_groups)
        state = defaultdict(dict)
        p_map = {
            old_p: p
            for old_p, p in zip(
                chain(*(g['params'] for g in old_groups)),
                chain(*(g['params'] for g in runner.optimizer.param_groups)))
        }
        for k, v in runner.optimizer.state.items():
            state[p_map[k]] = v
        runner.optimizer.state = state
        # convert model to fp16
        wrap_fp16_model(runner.model)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer `loss_scalar.py`

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
        scaled_loss.backward()
        # copy fp16 grads in the model to fp32 params in the optimizer

        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)

        has_overflow = self.loss_scaler.has_overflow(fp32_weights)
        # if has overflow, skip this iteration
        if not has_overflow:
            # scale the gradients back
            for param in fp32_weights:
                if param.grad is not None:
                    param.grad.div_(self.loss_scaler.loss_scale)
            if self.grad_clip is not None:
                self.clip_grads(fp32_weights)
            # update fp32 params
            runner.optimizer.step()
            # copy fp32 params to the fp16 model
            self.copy_params_to_fp16(runner.model, fp32_weights)
        self.loss_scaler.update_scale(has_overflow)
        if has_overflow:
            runner.logger.warning('Check overflow, downscale loss scale '
                                  f'to {self.loss_scaler.cur_scale}')
