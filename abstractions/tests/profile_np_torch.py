import numpy as np
import torch
import time

class CPUTimer:
    def __enter__(self):
        self.start = time.time()
        self.end = self.start
        self.duration = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.end = time.time()
            self.duration = self.end - self.start

class GPUTimer:
    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.duration = 1/1000.0 * self.start_event.elapsed_time(self.end_event)

# load into cache
x = np.random.standard_normal(size=(32, 4, 84, 84))

N = 1000

if torch.cuda.is_available():
    with CPUTimer() as t:
        x = np.random.standard_normal(size=(32, 4, 84, 84))
    print('CPUTimer:', t.duration)

    with GPUTimer() as t:
        x = np.random.standard_normal(size=(32, 4, 84, 84))
    print('GPUTimer:', t.duration)
    print()

with CPUTimer() as t:
    for _ in range(N):
        x_from_numpy = torch.from_numpy(x)
print('x_from_numpy: {}'.format(t.duration))
print(x_from_numpy.dtype)
print()

with CPUTimer() as t:
    for _ in range(N):
        x_as_tensor = torch.as_tensor(x)
print('x_as_tensor: {}'.format(t.duration))
print(x_as_tensor.dtype)
print()

with CPUTimer() as t:
    for _ in range(N):
        x_cast_then_as_tensor = torch.as_tensor(x, dtype=torch.float32)
print('x_cast_then_as_tensor: {}'.format(t.duration))
print(x_cast_then_as_tensor.dtype)
print()

with CPUTimer() as t:
    for _ in range(N):
        x_as_tensor_and_cast = torch.as_tensor(x.astype(np.float32))
print('x_as_tensor_and_cast: {}'.format(t.duration))
print(x_as_tensor_and_cast.dtype)
print()

with CPUTimer() as t:
    for _ in range(N):
        x_as_tensor_then_cast = torch.as_tensor(x.astype(np.float32)).float()
print('x_as_tensor_then_cast: {}'.format(t.duration))
print(x_as_tensor_then_cast.dtype)
print()

if torch.cuda.is_available():
    # load GPU
    x_cast_then_as_tensor_and_device = torch.as_tensor(x.astype(np.float32), device='cuda')

    with GPUTimer() as t:
        for _ in range(N):
            x_cast_then_as_tensor_and_device = torch.as_tensor(x.astype(np.float32), device='cuda')
    print('x_cast_then_as_tensor_and_device: {}'.format(t.duration))
    print(x_cast_then_as_tensor_and_device.dtype)
    print()

    with GPUTimer() as t:
        for _ in range(N):
            x_as_tensor_and_cast_and_device = torch.as_tensor(x, dtype=torch.float32, device='cuda')
    print('x_as_tensor_and_cast_and_device: {}'.format(t.duration))
    print(x_as_tensor_and_cast_and_device.dtype)
    print()

    with GPUTimer() as t:
        for _ in range(N):
            x_as_tensor_and_device_then_cast = torch.as_tensor(x, device='cuda').float()
    print('x_as_tensor_and_device_then_cast: {}'.format(t.duration))
    print(x_as_tensor_and_device_then_cast.dtype)
    print()

    with GPUTimer() as t:
        for _ in range(N):
            x_as_tensor_and_cast_then_device = torch.as_tensor(x, dtype=torch.float32).to(device='cuda')
    print('x_as_tensor_and_cast_then_device: {}'.format(t.duration))
    print(x_as_tensor_and_cast_then_device.dtype)
    print()

    with GPUTimer() as t:
        for _ in range(N):
            x_as_tensor_then_cast_then_device = torch.as_tensor(x).float().to(device='cuda')
    print('x_as_tensor_then_cast_then_device: {}'.format(t.duration))
    print(x_as_tensor_then_cast_then_device.dtype)
    print()

    with GPUTimer() as t:
        for _ in range(N):
            x_from_numpy_then_cast_then_device = torch.from_numpy(x).float().to(device='cuda')
    print('x_from_numpy_then_cast_then_device: {}'.format(t.duration))
    print(x_from_numpy_then_cast_then_device.dtype)
    print()
