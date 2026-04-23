
import os, time, datetime, torch, torch.distributed as dist
from collections import defaultdict, deque
from typing import Optional, List
from torch import Tensor
import torch.nn.functional as F

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors; self.mask = mask
    def to(self, device):
        return NestedTensor(self.tensors.to(device), self.mask.to(device) if self.mask is not None else None)
    def decompose(self): return self.tensors, self.mask
    def __repr__(self): return str(self.tensors)
    @property
    def shape(self): return self.tensors.shape
    @property
    def device(self): return self.tensors.device
    @property
    def dtype(self): return self.tensors.dtype

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = [3] + [max(img.shape[1] for img in tensor_list), max(img.shape[2] for img in tensor_list)]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        tensor = torch.zeros(batch_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=tensor_list[0].device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else: raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def collate_fn_crowd(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def accuracy(output, target, topk=(1,)):
    if target.numel() == 0: return [torch.zeros([], device=output.device)]
    maxk = max(topk); batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t(); correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type); device = parameters[0].grad.device
    return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2: return input_dict
    with torch.no_grad():
        names = []; values = []
        for k in sorted(input_dict.keys()): names.append(k); values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average: values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size); self.total = 0.0; self.count = 0; self.fmt = fmt
    def update(self, value, n=1): self.deque.append(value); self.count += n; self.total += value * n
    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized(): return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier(); dist.all_reduce(t); t = t.tolist()
        self.count = int(t[0]); self.total = t[1]
    @property
    def median(self): return torch.tensor(list(self.deque)).median().item()
    @property
    def avg(self): return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()
    @property
    def global_avg(self): return self.total / self.count
    @property
    def max(self): return max(self.deque)
    @property
    def value(self): return self.deque[-1]
    def __str__(self): return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"): self.meters = defaultdict(SmoothedValue); self.delimiter = delimiter
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor): v = v.item()
            assert isinstance(v, (float, int)); self.meters[k].update(v)
    def __getattr__(self, attr):
        if attr in self.meters: return self.meters[attr]
        if attr in self.__dict__: return self.__dict__[attr]
        raise AttributeError("'MetricLogger' object has no attribute '{}'".format(attr))
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items(): loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)
    def synchronize_between_processes(self):
        for meter in self.meters.values(): meter.synchronize_between_processes()
    def add_meter(self, name, meter): self.meters[name] = meter
    def log_every(self, iterable, print_freq, header=''):
        i = 0; start_time = time.time(); end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}'); data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}'])
        for obj in iterable:
            data_time.update(time.time() - end); yield obj; iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)))
            i += 1; end = time.time()
        total_time = time.time() - start_time
        print('{} Total time: {} ({:.4f} s / it)'.format(header, str(datetime.timedelta(seconds=int(total_time))), total_time / len(iterable)))

def is_dist_avail_and_initialized():
    if not dist.is_available(): return False
    if not dist.is_initialized(): return False
    return True
def get_world_size(): return dist.get_world_size() if is_dist_avail_and_initialized() else 1
def get_rank(): return dist.get_rank() if is_dist_avail_and_initialized() else 0
def is_main_process(): return get_rank() == 0
def save_on_master(*args, **kwargs): 
    if is_main_process(): torch.save(*args, **kwargs)
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'gloo' # FORCE GLOO
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend="gloo", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force: builtin_print(*args, **kwargs)
    __builtin__.print = print
