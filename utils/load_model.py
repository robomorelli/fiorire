# utils/load_model.py

from utils.model_registry import MODEL_REGISTRY

def get_model(cfg, **kwargs):
    model_name = cfg.model.name
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    model = model_class(cfg, **kwargs)
    print(f'Loading model: {model_name}')
    print(model)
    return model
