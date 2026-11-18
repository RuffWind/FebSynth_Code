
from .base import BaseSyntheticDataGenerator
from .copula import CopulaGenerator
from .ctgan import CTGANGenerator
from .tvae import TVAEGenerator
from .taddpm import TabDDPMGenerator

__all__ = [
    'BaseSyntheticDataGenerator',
    'CopulaGenerator',
    'CTGANGenerator',
    'TVAEGenerator',
    'TabDDPMGenerator',
]

GENERATOR_REGISTRY = {
    'copula': CopulaGenerator,
    'ctgan': CTGANGenerator,
    'tvae': TVAEGenerator,
    'tabddpm': TabDDPMGenerator,
}

def get_generator(name: str, **kwargs):
    name_lower = name.lower()

    if name_lower not in GENERATOR_REGISTRY:
        available = ', '.join(GENERATOR_REGISTRY.keys())
        raise ValueError(
            f"未知的生成器: {name}\n"
            f"可用的生成器: {available}"
        )

    generator_class = GENERATOR_REGISTRY[name_lower]
    return generator_class(**kwargs)

def list_generators():
    print("\n可用的生成器:")
    print("="*80)
    for name in sorted(set(GENERATOR_REGISTRY.keys())):
        generator_class = GENERATOR_REGISTRY[name]
        doc = generator_class.__doc__.strip().split('\n')[0] if generator_class.__doc__ else "无描述"
        print(f"  {name:20s} - {doc}")
    print("="*80)
