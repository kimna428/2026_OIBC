from .convnext1d import ConvNeXt1D
from .weather_transformer import WeatherTransformer
from .cross_site_attention import CrossSiteAttention
from .common import model_inputs

MODELS={'cnn':ConvNeXt1D,'transformer':WeatherTransformer,'cross_site':CrossSiteAttention}
def build_model(config,dims):
    params=dict(config); name=params.pop('name')
    return MODELS[name](**dims,**params)
def parameter_count(model): return sum(p.numel() for p in model.parameters())
