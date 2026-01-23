from copy import deepcopy

def inject_training_hparams_from_ckpt(cfg, ckpt_dict):
    """
    Prende dal checkpoint gli iperparametri originali di training (lr, batch_size,
    lr_patience, es_patience) e li inietta in cfg.opt.

    cfg: oggetto OmegaConf (dal config YAML corrente)
    ckpt_dict: dict caricato dal checkpoint (.pt)
    """
    cfg = deepcopy(cfg)

    # estrai cfg.opt dal checkpoint
    ckpt_cfg = ckpt_dict.get("cfg", None)
    if ckpt_cfg is None:
        raise ValueError("Il checkpoint non contiene la configurazione 'cfg'.")

    ckpt_opt = ckpt_cfg.get("opt", {})
    if not ckpt_opt:
        raise ValueError("Il checkpoint non contiene la sezione 'cfg.opt'.")

    # parametri da estrarre
    keys_to_copy = ["lr", "batch_size", "lr_patience", "es_patience"]

    for key in keys_to_copy:
        if key in ckpt_opt:
            cfg.opt[key] = ckpt_opt[key]

    return cfg
