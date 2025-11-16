from typing import Literal
from pydantic import (BaseModel, Field, ConfigDict)

class Cpu(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_cores: int = Field(..., ge=0)
    cpu_threads: int = Field(..., ge=0)
    cpu_frq: float = Field(..., ge=0.0)
    cpu_clock: float = Field(..., ge=0.0)
    cpu_multiplier: int = Field(..., ge=0)
    cpu_tdp: int = Field(..., ge=0)
    cpu_prcss: int = Field(..., ge=0)
    cpu_die: int = Field(..., ge=0)
    cpu_has_oc: Literal["Y", "N"]