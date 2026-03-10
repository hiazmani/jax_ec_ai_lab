from importlib import import_module

_MECHS = {
    "no_exchange": "no_exchange.NoExchange",
    "midpoint": "midpoint.MidpointPrice",
    "double_auction": "double_auction.DoubleAuction",
    "agent_pricing": "agent_pricing.CentralAgentPricing",
}


def get_mechanism(name: str, **kwargs):
    """Returns an exchange mechanism instance based on the given name.

    Args:
        name (str): The name of the exchange mechanism. Must be one of:
            'no_exchange', 'midpoint', 'double_auction'.
        **kwargs: Additional keyword arguments to pass to the mechanism's constructor.

    Returns:
        An instance of the specified exchange mechanism.
    """
    if name not in _MECHS:
        raise ValueError(f"Unknown mechanism {name}")
    module_path, cls_name = _MECHS[name].rsplit(".", 1)
    cls = getattr(import_module(f".{module_path}", __name__), cls_name)
    return cls(**kwargs)
