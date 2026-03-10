"""Unit tests for the battery management module in jax_ec."""

import jax.numpy as jnp

from jax_ec.environment.battery import (
    charge_battery,
    create_battery,
    get_default_battery_cfg,
    get_disabled_battery_cfg,
)


def test_create_battery_from_cfg(debug=False):
    """Test creating a battery from configuration.

    Args:
        debug: If True, print debug information.

    Returns:
        None
    """
    cfg = get_default_battery_cfg()
    battery = create_battery(**cfg)
    for key in cfg:
        assert jnp.isclose(getattr(battery, key), cfg[key])


def test_battery_charge_discharge(debug=False):
    """Test battery charging and discharging behavior.

    Args:
        debug: If True, print debug information.

    Returns:
        None
    """
    cfg = get_default_battery_cfg()
    if debug:
        print("\n== BATTERY CHARGE/DISCHARGE ==")
    init_battery = create_battery(**cfg)
    if debug:
        print(f"    * Initial Battery SoC: {init_battery.soc}")

    # Charge the battery
    battery_charge, charge_eff = charge_battery(init_battery, -2.0)
    if debug:
        print(f"\t* Battery SoC after charging: {battery_charge.soc}")
        print(f"\t* Efficiency: {charge_eff}")
    assert battery_charge.soc >= init_battery.soc
    assert charge_eff == -2.0

    # Discharge the battery
    battery_discharge, discharge_eff = charge_battery(init_battery, 1.0)
    if debug:
        print(f"\t* Battery SoC after discharging: {battery_discharge.soc}")
        print(f"\t* Efficiency: {charge_eff}")
    assert battery_discharge.soc <= init_battery.soc
    assert discharge_eff == 1.0


def test_battery_limits(debug=False):
    """Test battery limits during charge and discharge.

    Args:
        debug: If True, print debug information.

    Returns:
        None
    """
    cfg = get_default_battery_cfg()
    if debug:
        print("\n== BATTERY LIMITS ==")

    # Try charging an already charged battery
    cfg["soc"] = 1.0  # Set to full charge
    full_battery = create_battery(**cfg)
    if debug:
        print(f"    * Full Battery SoC: {full_battery.soc}")
    # Attempt to charge beyond full capacity
    battery_overcharge, overcharge_eff = charge_battery(full_battery, -2.0)
    if debug:
        print(f"\t* Battery SoC after overcharging: {battery_overcharge.soc}")
        print(f"\t* Effective Energy: {overcharge_eff}")
    assert battery_overcharge.soc <= 1.0
    assert jnp.isclose(overcharge_eff, 0.0)

    # Try discharging an already empty battery
    cfg = get_default_battery_cfg()
    cfg["soc"] = 0.0  # Set to empty
    empty_battery = create_battery(**cfg)
    if debug:
        print("  => DISCHARGE")
        print(f"    * Empty Battery SoC: {empty_battery.soc}")
    # Attempt to discharge below empty capacity
    battery_overdischarge, overdischarge_eff = charge_battery(empty_battery, 2.0)
    if debug:
        print(f"\t* Battery SoC after discharging: {battery_overdischarge.soc}")
        print(f"\t* Effective Energy: {overdischarge_eff}")
    assert battery_overdischarge.soc >= 0.0
    assert jnp.isclose(overdischarge_eff, 0.0)


def test_battery_degradation(debug=False):
    """Test battery degradation over multiple charge/discharge cycles.

    Args:
        debug: If True, print debug information.

    Returns:
        None
    """
    cfg = get_default_battery_cfg()
    if debug:
        print("\n== BATTERY DEGRADATION ==")
    battery = create_battery(**cfg)

    # Simulate charge/discharge cycles
    for _ in range(10):
        battery, _ = charge_battery(battery, -1.0)  # charge
        battery, _ = charge_battery(battery, 1.0)  # discharge

    expected_capacity = cfg["capacity"] - 20 * cfg["capacity_loss_coefficient"]
    if debug:
        print(f"\t* Expected Capacity after cycles: {expected_capacity}")
        print(f"\t* Actual Capacity after cycles: {battery.capacity}")
    assert jnp.isclose(battery.capacity, expected_capacity, atol=1e-4)


def test_disabled_battery(debug=False):
    """Test behavior of a disabled battery.

    Args:
        debug: If True, print debug information.

    Returns:
        None
    """
    if debug:
        print("\n== DISABLED BATTERY ==")
    cfg = get_disabled_battery_cfg()
    battery = create_battery(**cfg)
    if debug:
        print(f"    * Disabled Battery SoC: {battery.soc}")
        print(f"    * Disabled Battery Capacity: {battery.capacity}")

    # Attempt to charge/discharge
    battery_after_charge, charge_eff = charge_battery(battery, -1.0)
    if debug:
        print(f"\t* Battery SoC after charging: {battery_after_charge.soc}")
        print(f"\t* Efficiency: {charge_eff}")
    assert battery_after_charge.soc == 0.0
    assert charge_eff == 0.0

    battery_after_discharge, discharge_eff = charge_battery(battery, 1.0)
    if debug:
        print(f"\t* Battery SoC after discharging: {battery_after_discharge.soc}")
        print(f"\t* Efficiency: {discharge_eff}")
    assert battery_after_discharge.soc == 0.0
    assert discharge_eff == 0.0


def test_residual_energy_transfer():
    """Test residual energy transfer during charge/discharge cycles."""
    cfg = get_default_battery_cfg()
    battery = create_battery(**cfg)

    # Discharge 1 kWh
    battery2, eff = charge_battery(battery, 1.0)
    assert eff > 0  # net flow to grid

    # Charge 1 kWh
    battery3, eff2 = charge_battery(battery2, -1.0)
    assert eff2 < 0  # net flow from grid


def test_soc_after_cycles():
    """Test that SoC returns to near original after full charge/discharge cycles."""
    cfg = get_default_battery_cfg()
    cfg["round_trip_efficiency"] = 1.0
    cfg["soc"] = 0.0

    battery = create_battery(**cfg)

    steps = 5
    for _ in range(steps):
        battery, _ = charge_battery(battery, -1.0)

    for _ in range(steps):
        battery, _ = charge_battery(battery, 1.0)

    # Should return near original SoC (minus tiny losses)
    assert jnp.isclose(battery.soc, cfg["soc"], atol=0.05)


if __name__ == "__main__":
    test_create_battery_from_cfg()
    test_battery_charge_discharge()
    test_battery_limits()
    test_battery_degradation()
    test_disabled_battery()
    test_residual_energy_transfer()
