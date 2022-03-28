import torch
import numpy as np
import rasterio as rio
from pathlib import Path
from torchsparse.nn import Conv3d
from torchsparse import SparseTensor
from typing import Tuple, List, Union


def mean_conv(
    t: SparseTensor,
    kernel_size: Union[int, Tuple[int, int, int], List[int]] = 3
):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, ] * 3
    conv = Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=(1, 1, 1),
        dilation=1,
        bias=False,
        transposed=False
    ).to(device=t.F.device)
    w = 1 / torch.prod(
        torch.tensor(kernel_size, dtype=torch.float, device=t.F.device)
    ).item()
    torch.nn.init.constant_(conv.kernel, val=w)
    out = conv(t)
    return out.detach()


def calc_roughness(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    kernel_size: Union[int, Tuple[int, int], List[int]] = 3,
    device: torch.device = torch.device('cpu')
):
    try:
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size, 1
        elif isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) == 2:
                kernel_size = *kernel_size, 1
            elif len(kernel_size) == 1:
                kernel_size = *(kernel_size * 2), 1
            else:
                raise ValueError(
                    f"Incompatible 'kernel_size': {kernel_size}. " +
                    "Expected <int> or <int, int> "
                )
        else:
            raise ValueError(f"Illegal 'kernel_size': {kernel_size}")

        with rio.open(src_path, 'r') as src:
            meta = src.meta.copy()
            img = np.expand_dims(
                a=np.moveaxis(
                    a=src.read(masked=True),
                    source=0,
                    destination=-1
                ),
                axis=-2
            )  # X-Y-Z-B

        indexes = np.moveaxis(
            a=np.indices(
                dimensions=img.shape,
                dtype=np.int64,
                sparse=False
            ),
            source=0,
            destination=-1
        ).reshape(-1, len(img.shape))
        indexes = indexes[np.logical_not(img.mask.ravel()), slice(None)]
        # noinspection PyUnresolvedReferences
        features = img.compressed().reshape(-1, 1)

        assert indexes.shape[0] == features.shape[0]

        indexes = torch.tensor(indexes, dtype=torch.int, device=device)
        features = torch.tensor(features, dtype=torch.float, device=device)

        st = SparseTensor(coords=indexes, feats=features)
        mu = mean_conv(t=st, kernel_size=kernel_size)
        st.F = (st.F - mu.F) ** 2.0
        st = mean_conv(t=st, kernel_size=kernel_size)
        idx_list = [
            st.C[slice(None), i].to(dtype=torch.long)
            for i in range(st.C.size(-1))
        ]
        values = torch.squeeze(input=st.F)

        out = torch.full(
            size=img.shape,
            fill_value=np.nan,
            dtype=st.F.dtype,
            device=st.F.device
        )
        out[idx_list] = values
        out = torch.squeeze(input=out, dim=-2)
        out = torch.permute(input=out, dims=(-1, 0, 1))

        out = out.cpu().numpy()
        meta['dtype'] = out.dtype
        meta['nodata'] = np.nan
        with rio.open(dst_path, "w", **meta) as dst:
            dst.write(out)
        return True
    except Exception as err:
        print(err)
        return False


if __name__ == "__main__":
    import argparse
    from halo import Halo

    parser = argparse.ArgumentParser(
        description=(
            'Calculate surface roughness / localized standard deviation ' +
            'from sparse image'
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
        type=int,
        nargs='*',
        required=False,
        dest='ks',
        default=(3, 3),
        help='Specify kernel size. Hint: int or int int'
    )
    parser.add_argument(
        '-d', '--device',
        metavar='Target device',
        action='store',
        type=str,
        default='auto',
        required=False,
        dest='dev',
        help="Specify target device. Default: 'cpu'"
    )
    args = parser.parse_args()
    srcpath = Path(args.import_path)
    dstpath = Path(args.export_path)
    if args.dev.lower() == 'auto':
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
    else:
        dev = torch.device(args.dev)
    with Halo(
        spinner='dots',
        text="Processing ...",
        color="yellow",
        text_color='grey'
    ) as spinner:
        status = calc_roughness(
            src_path=srcpath,
            dst_path=dstpath,
            kernel_size=args.ks,
            device=dev
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
