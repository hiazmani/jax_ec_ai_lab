# JAXEC: JAX-based Energy Communities Simulator

## Grid Charging Behavior
The simulator supports two types of battery charging modes:
* **Local-only (default)**: Batteries can only charge using *local energy surplus*. This models self-consumption behavior where batteries are only charged from local PV/production, not from the grid.
* **Grid-Enabled**: Batteries can also charge using *energy from the grid* if requested by the agent.

This is controlled via the allow_grid_charging flag in the environment:
```python
env = JAXECEnvironment(..., allow_grid_charging=True)
```

The reason for this design decision is that whether charging from the grid is allowed or not heavily depends on the context:
* In many real-world residential setups charging from the grid is not possible/allowed:
    * Many inverters and battery management systems are designed to only allow charging from solar.
    * Some countries even prohibit charging residential batteries from the grid (e.g., Some incentives in the USA, such as the SMART incentive program and the Federal Investment Tax credit (ITC) have limitations on charging from the grid[^1]; Ofgem’s FIT/SEG rules in the UK likewise exclude batteries charged from grid energy[^2])
* But in research contexts:
    * Grid charging enables strategic behavior (e.g., arbitrage, dynamic pricing)
    * It’s essential for modeling smart grid participation or time-of-use optimization

Therefore both scenarios and supported and can be toggled using the `allow_grid_charging` variable.

When `allow_grid_charging=False`, battery charging is limited to local production surplus. This ensures that agents cannot charge their batteries with unavailable (grid) energy. If grid charging is enabled, batteries can draw from the grid and will incur cost accordingly.

## References
[^1]: National Grid US. (2025). Battery Program. https://www.nationalgridus.com/MA-Home/Connected-Solutions/BatteryProgram
[^2]: Ofgem. (2018). Guidance for generators: Co-location of electricity storage facilities with renewable generation supported under the Renewables Obligation or Feed-in Tariff schemes (Version 1). Office of Gas and Electricity Markets. https://www.ofgem.gov.uk/sites/default/files/docs/2018/06/final_storage_guidance_0.pdf
