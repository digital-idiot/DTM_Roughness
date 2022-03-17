 
## Efficiently Calculate **Surface Roughness** or **Localized RMSE**

The surface roughness / localized RMSE is derived from a specified image or DTM / DEM. Image can also multi-channel, in that case roughness will be calculated channel wise. 

```
Usage: roughness.py [-h] -i Input file path -o Output file path [-k [Kernel size ...]] [-p Padding mode] [-d Target device]

Calculate surface roughness / localized RMSE

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