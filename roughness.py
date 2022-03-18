import torch
import argparse
import numpy as np
from halo import Halo
import rasterio as rio
from pathlib import Path
from torch import nn as tnn
from typing import Tuple, Union
from torch.nn.functional import pad as torch_pad


class DynamicScalePad2D(tnn.Module):
    def __init__(
        self,
        mode: str,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        *args, ** kwargs
    ) -> None:
        super(DynamicScalePad2D, self).__init__()
        if not(mode in {'constant', 'reflect', 'replicate' or 'circular'}):
            raise NotImplementedError(
                "Unknown padding mode: {}!".format(str(mode))
            )
        else:
            self.mode = mode

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        assert isinstance(
            kernel_size, (tuple, list)
        ) and len(kernel_size) == 2 and all(
            isinstance(n, int) and n > 0
            for n in kernel_size
        ), "Invalid kernel_size: {}".format(kernel_size)
        self.ks = kernel_size

        if isinstance(stride, int):
            stride = (stride,) * 2
        assert isinstance(
            stride, (tuple, list)
        ) and len(stride) == 2 and all(
            isinstance(m, int) and m > 0
            for m in stride
        ), "Invalid stride: {}".format(stride)
        self.st = stride

        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise NotImplementedError(
                '{}-dimensional tensor is not supported!'.format(x.ndim)
            )

        sp_shape = x.shape[-2:]
        pads = list()
        for i in reversed(range(2)):
            res = sp_shape[i] % self.st[i]
            s = self.st[i] if res == 0 else res
            bi_pad = max(0, (self.ks[i] - s))
            pad_1 = bi_pad // 2
            pad_2 = bi_pad - pad_1
            pads.append(pad_1)
            pads.append(pad_2)
        return torch_pad(
            input=x,
            pad=pads,
            mode=self.mode,
            *self.args,
            **self.kwargs
        )

def convolute(
    tensor,
    kernel_size=(3, 3),
    padding_mode='replicate',
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    groups = tensor.size(1)
    pad = DynamicScalePad2D(
        mode=padding_mode,
        kernel_size=kernel_size,
        stride=(1, 1)
    )
    conv = tnn.Conv2d(
        in_channels=groups,
        out_channels=groups,
        kernel_size=kernel_size,
        stride=(1, 1),
        padding=0,
        dilation=1,
        groups=groups,
        bias=False,
        padding_mode='replicate',
        device=tensor.device
    )
    w = 1 / torch.prod(
        torch.tensor(kernel_size, dtype=torch.float, device=tensor.device)
    ).item()

    tnn.init.constant_(conv.weight, val=w)
    tensor = pad(tensor)
    tensor = conv(tensor)
    return tensor

def calculate_sigma(tensor, kernel_size=(3, 3), padding_mode='replicate'):
    tensor = tensor.to(dtype=torch.float)
    mu = convolute(
        tensor=tensor,
        kernel_size=kernel_size,
        padding_mode=padding_mode
    )
    var = (tensor - mu) ** 2
    sigma = convolute(
        tensor=var,
        kernel_size=kernel_size,
        padding_mode=padding_mode
    ) ** 0.5
    return sigma


def process(
    src_path,
    dst_path,
    kernel_size=(3, 3),
    padding_mode='replicate',
    device=torch.device('cpu')
):
    try:
        with rio.open(src_path, "r") as src:
            meta = src.meta.copy()
            img = src.read(masked=True)
            nd_mask = img.mask
        img.fill_value = 0
        img = img.filled()
        tensor = torch.from_numpy(img.astype(np.float32)).to(
            dtype=torch.float,
            device=device
        )

        tensor = torch.unsqueeze(tensor, 0)
        sigma = calculate_sigma(
            tensor=tensor,
            kernel_size=kernel_size,
            padding_mode=padding_mode
        )
        sigma = torch.squeeze(sigma, 0)
        dst_img = sigma.detach().cpu().numpy()
        meta['dtype'] = dst_img.dtype

        with rio.open(dst_path, "w", **meta) as dst:
            dst_img = np.ma.masked_array(
                dst_img, nd_mask, fill_value=meta['nodata']
            )
            dst_img = dst_img.filled()
            dst.write(dst_img)
            return True
    except Exception as exc:
        print(exc)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Calculate surface roughness / localized RMSE'
        )
    )
    parser.add_argument(
        '-i', '--input',
        metavar='Input file path',
        action='store',
        type=str,
        required=True,
        dest='import_path',
        help='Specify input file'
    )
    parser.add_argument(
        '-o', '--output',
        metavar='Output file path',
        action='store',
        type=str,
        required=True,
        dest='export_path',
        help='Specify output file'
    )
    parser.add_argument(
        '-k', '--kernel',
        metavar='Kernel size',
        action='store',
        nargs='*',
        required=False,
        dest='ks',
        default=(3, 3),
        help='Specify kernel size. Hint: int or int int'
    )
    parser.add_argument(
        '-p', '--padding',
        metavar='Padding mode',
        action='store',
        type=str,
        default='replicate',
        required=False,
        dest='pm',
        help=(
            "Specify padding mode. "+
            "Choices: 'zeros', 'reflect', 'replicate' or 'circular'. " +
            "Default: 'replicate'"
        )
    )
    parser.add_argument(
        '-d', '--device',
        metavar='Target device',
        action='store',
        type=str,
        default='cpu',
        required=False,
        dest='dev',
        help="Specify target device. Default: 'cpu'"
    )
    args = parser.parse_args()
    src_path = Path(args.import_path)
    dst_path=Path(args.export_path)
    if dst_path.suffix != src_path.suffix:
        dst_path = dst_path.parent / (dst_path.stem + src_path.suffix)
    kernel_size = tuple([int(i) for i in args.ks])

    with Halo(
            spinner='dots',
            text="Processing",
            color="yellow",
            text_color='grey'
        ) as spinner:
        status = process(
            src_path=src_path,
            dst_path=dst_path,
            kernel_size=kernel_size,
            padding_mode=args.pm,
            device=torch.device(args.dev)
        )
        if status:
            spinner.stop_and_persist(
                symbol=u'✅',
                text="Process completed successfully!"
            )
            spinner.text_color = 'cyan'
        else:
            spinner.stop_and_persist(
                symbol=u'❌',
                text="Process Failed!"
            )
            spinner.text_color = 'magenta'
