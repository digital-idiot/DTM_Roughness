 
## Efficiently Calculate **Surface Roughness** or **Localized RMSE**

The surface roughness / localized standard deviation (<img src="https://render.githubusercontent.com/render/math?math=\sigma_{kernel}">) is derived from a specified image or DTM / DEM. Image can also multi-channel, in that case roughness will be calculated channel wise.

* Math:
<img src="https://render.githubusercontent.com/render/math?math=\sigma_i = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \left( x_i - \frac{1}{N}\sum_{i=1}^{N} x_i \right)}">

where <img src="https://render.githubusercontent.com/render/math?math=N"> is the number of elements in the specified kernel.

```
Usage: roughness.py [-h] -i Input file path -o Output file path [-k [Kernel size ...]] [-p Padding mode] [-d Target device]

Calculate surface roughness / localized standard deviation

optional arguments:
  -h, --help            show this help message and exit
  -i Input file path, --input Input file path
                        Specify input file
  -o Output file path, --output Output file path
                        Specify output file
  -k [Kernel size ...], --kernel [Kernel size ...]
                        Specify kernel size. Hint: int or int int
  -p Padding mode, --padding Padding mode
                        Specify padding mode. Choices: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'replicate'
  -d Target device, --device Target device
                        Specify target device. Default: 'cpu'
``` 