"""Battery configuration utilities."""


def battery_cfg_compat(cfg: dict) -> dict:
    """Rename keys so env.reset() sees `soc` (compat with `initial_soc`)."""
    cfg = dict(cfg)
    if "soc" not in cfg and "initial_soc" in cfg:
        cfg["soc"] = cfg.pop("initial_soc")
    return cfg


def make_battery_cfgs(agent_ids, base_cfg: dict) -> dict:
    """Clone a base battery cfg for each agent id."""
    base = battery_cfg_compat(base_cfg)
    return {aid: dict(base) for aid in agent_ids}
