import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def build_baseline(model_name: str, in_channels: int = 3, out_channels: int = 3):
    if model_name == "fukami_cnn":
        from fukami2019_cnn import Fukami2019_CNN
        return Fukami2019_CNN(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "fukami_dscms":
        from fukami2019_dscms import Fukami2019_DSCMS
        return Fukami2019_DSCMS(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "guastoni":
        from guastoni2021_fcn import Guastoni2021_FCN
        return Guastoni2021_FCN(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "yousif":
        from yousif2021_msesrgan import Yousif2021_MSESRGANGenerator
        return Yousif2021_MSESRGANGenerator(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "kim":
        from kimlee2021_cyclegan import KimLee2021_CycleGANGenerator
        return KimLee2021_CycleGANGenerator(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
