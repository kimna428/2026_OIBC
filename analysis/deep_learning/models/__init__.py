from .convnext1d import ConvNeXt1D
from .weather_transformer import WeatherTransformer
from .cross_site_attention import CrossSiteAttention
from .common import model_inputs

MODELS={'cnn':ConvNeXt1D,'transformer':WeatherTransformer,'cross_site':CrossSiteAttention}
def build_model(config,dims):
    params=dict(config); name=params.pop('name')
    import inspect
    allowed=set(inspect.signature(MODELS[name].__init__).parameters)-{'self','unused'}
    if set(params)-allowed: raise ValueError(f'Unknown architecture parameters: {set(params)-allowed}')
    return MODELS[name](**dims,**params)
def parameter_count(model): return sum(p.numel() for p in model.parameters())
