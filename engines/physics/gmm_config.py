from pydantic import BaseModel, Field
from typing import Literal

class GMMConfig(BaseModel):
    K_target: int = 12
    K_min: int = 8
    K_max: int = 20
    w_min: float = 0.0025              # 0.25%
    merge_kl_thresh: float = 0.02      # small = merge aggressively
    sigma_mag_ticks: float = 2.0       # narrow magnet component
    magnet_spawn_weight: float = 0.02  # 2% initial weight each
    max_magnets: int = 4

class DynamicsConfig(BaseModel):
    a_I_over_beta: float = 1.0         # drift coefficient on I/beta
    b_charm: float = 0.5               # drift coefficient on charm tilt
    c_wyckoff: float = 0.75            # drift coefficient on wyckoff trend
    q0: float = 0.02                   # diffusion floor (price units; tune per symbol)
    gex_short_gamma_mult: float = 1.6  # diffusion upshift when GEX < 0
    gex_long_gamma_mult: float = 0.7   # diffusion downshift when GEX > 0

class FieldConfig(BaseModel):
    lambda_liq: float = 1.0            # liquidity well strength
    lambda_strike: float = 1.0         # strike well strength (signed via GEX)
    lambda_wyck: float = 0.8           # wyckoff barrier strength
    tau0: float = 1.0                  # base temperature
    tau_short_gamma_mult: float = 1.5
    tau_long_gamma_mult: float = 0.8

class UniverseConfig(BaseModel):
    prefilter_M: int = 60
    active_N: int = 15
    min_hold_minutes: int = 5
    add_margin: float = 0.15           # new entrant must exceed by 15%
    drop_threshold_frac: float = 0.60  # drop if <60% of cutoff
    max_turnover_per_minute: int = 3

class CostConfig(BaseModel):
    spread_mult: float = 0.8           # fraction of spread you “pay” (depends on crossing)
    fee_per_share: float = 0.0         # set if applicable
    slippage_mult_rv: float = 0.1      # extra cost proportional to rv

class ModelConfig(BaseModel):
    cadence: Literal["1m"] = "1m"
    gmm: GMMConfig = Field(default_factory=GMMConfig)
    dyn: DynamicsConfig = Field(default_factory=DynamicsConfig)
    field: FieldConfig = Field(default_factory=FieldConfig)
    uni: UniverseConfig = Field(default_factory=UniverseConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
