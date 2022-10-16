"""Constants."""
from __future__ import annotations

import pandas as pd

MTDF = pd.DataFrame()
COLOR_MAP = {
    "Gas CC": "#c85c19",
    "Gas CT": "#f58228",
    "Gas RICE": "#fbbb7d",
    "Gas ST": "#ffdaab",
    "Coal": "#5f2803",
    "Other Fossil": "#7d492c",
    "Biomass": "#556940",
    "Solar": "#ffcb05",
    "Onshore Wind": "#005d7f",
    "Offshore Wind": "#529cba",
    "Storage": "#7b76ad",
    "Charge": "#7b76ad",
    "Discharge": "#7b76ad",
    "Curtailment": "#eec7b7",
    "Deficit": "#df897b",
    "Net Load": "#58585b",
    "Grossed Load": "#58585b",
}
PLOT_MAP = {
    "Petroleum Liquids": "Other Fossil",
    "Natural Gas Steam Turbine": "Gas ST",
    "Conventional Steam Coal": "Coal",
    "Natural Gas Fired Combined Cycle": "Gas CC",
    "Natural Gas Fired Combustion Turbine": "Gas CT",
    "Natural Gas Internal Combustion Engine": "Gas RICE",
    "Coal Integrated Gasification Combined Cycle": "Coal",
    "Other Gases": "Other Fossil",
    "Petroleum Coke": "Other Fossil",
    "Wood/Wood Waste Biomass": "Biomass",
    "Other Waste Biomass": "Biomass",
    "Landfill Gas": "Biomass",
    "Municipal Solid Waste": "Biomass",
    "All Other": "Other Fossil",
    "solar": "Solar",
    "onshore_wind": "Onshore Wind",
    "offshore_wind": "Offshore Wind",
    "curtailment": "Curtailment",
    "deficit": "Deficit",
    "charge": "Storage",
    "discharge": "Storage",
}
