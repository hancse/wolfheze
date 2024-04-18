# In[]:

import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5 backend

network = pypsa.Network()

# In[Buses]
network.add("Bus", "AC_bus", carrier='AC')  # ....Main AC_bus
network.add("Bus", "40C_bus", carrier='Heat_40')  # ,carrier = "heat") #....40 degree celcius heat bus
network.add("Bus", "60C_bus", carrier='Heat_60')  # ,carrier = "heat") #....60 degree celcius heat bus
network.add("Bus", "H2_bus", carrier='H2')  # ,carrier = "gas")   #....Hydrogen Bus
network.add("Bus", "48VDC_bus", carrier='DC')  # 48V box inside PowerBox
network.add("Bus", "House_bus", carrier='AC')  # 8 houses bus

# Set the simulation snapshot times
snapshots = pd.date_range('2023-01-01 00:00', periods=4, freq='H')
network.set_snapshots(snapshots)

# In[Generators]


network.add("Generator",
            "PV_panels",
            bus="AC_bus",
            carrier='AC',
            p_nom=51.8,
            p_min_pu=0,
            p_max_pu=0.8,
            marginal_cost=0.2)  # ...Power from PV panels

# In[Generators]

network.add("Generator",
            "Grid_gen",
            bus="AC_bus",
            carrier='AC',
            p_nom=55.2,
            marginal_cost=np.array([-0.2, -0.25, -0.2, -0.2]))  # power from grid bus

# In[Load 1]

network.add("Load",
            "Electric_load",
            bus="House_bus",
            p_set=np.array([8, 5, 15, 10]))  # Electric load connected to the house bus

# In[Load 2]

network.add("Load",
            "space_heating",
            bus="40C_bus",
            carrier='Heat_40',
            p_set=40)  # space heating to 40C for 8 houses, peak load

# In[Load 3]

network.add("Load",
            "EV_chargers",
            bus="House_bus",
            p_set=30)  # 1 charger per house 8 in total

# In[Load 4]

network.add("Load",
            "DHW_demand",
            bus="60C_bus",
            carrier='Heat_60',
            p_set=2)  # DHW_demand added to 60C bus for 8houses

# In[Link 1]

network.add("Link",
            "Electrolyzer",
            bus0="AC_bus",
            bus1="H2_bus",
            p_nom=5,
            efficiency=1)  # Efficiency is not known yet, has to be checked in Datasheet!

# In[Link 2]

network.add("Link",
            "Inverter",
            bus0="48VDC_bus",
            bus1="AC_bus",
            p_nom=5,
            efficiency=1)  # Efficiency is not known.

# In[Link 3]

network.add("Link",
            "Feulcell",
            bus0="H2_bus",
            bus1="48VDC_bus",
            p_nom=5,
            efficiency=1)  # Efficiency is not known.

# In[Link 4]

network.add("Link",
            "Heat pump",
            bus0="AC_bus",
            bus1="40C_bus",
            p_nom=20,
            efficiency=3)

# In[Link 5]

network.add("Link",
            "Booster Heat Pump",
            bus0="AC_bus",
            bus1="40C_bus",
            bus2="60C_bus",
            p_nom=0.5,
            p_nom_max=0.5,
            efficiency=-3,
            efficiency2=4)

# In[Link 6]

network.add("Link",
            "Booster Electric Element",
            bus0="AC_bus",
            bus1="60C_bus",
            p_nom=2,
            efficiency=1)

# In[Link 7]

network.add("Link",
            "GridFeedIn",
            bus0="Grid_gen",
            bus1="AC_bus",
            p_nom=55.2,
            efficiency=-1,
            marginal_cost=0.2)

# In[Link 8]

network.add("Link",
            "HouseConnection",
            bus0="AC_bus",
            bus1="House_bus",
            effiency=-1,
            p_nom=138,
            p_nom_max=138)

# In[Store 1]
"""
network.add("Store",
            "H2_tanks,",
            bus="H2_bus",
            carrier="H2",
            e_nom=1200,
            e_initial=600,
            e_nom_opt=10,
            p_set=5,
            e_cyclic=True)
"""
# In[Store 2]

# network.add("Store",
#            "Buffertank",
#            bus = "40C_bus",
#            carrier = "Heat_40",
#            e_nom = 23.2,
#            e_initial = 10,
#            e_min_pu = 0.2,
#            p_set = 6,
#            e_cyclic = False )


# In[Store 3]

network.add("Store",
            "Battery",
            bus="48VDC_bus",
            e_nom=30,  # Nominal energy capacity in MWh
            e_initial=15,  # Initial energy stored in MWh
            e_min_pu=0,  # Minimum state of charge as per unit of e_nom
            e_max_pu=1.0,  # Maximum state of charge as per unit of e_nom
            p_set = 10,
            marginal_cost=0.1)  # Marginal cost per MWh of stored energy

# In[]
network.optimize()


# Perform a Linear Optimal Power Flow to balance generation and load
network.lopf(network.snapshots)

# Print the results
print("Generator Output PV panels:", network.generators_t.p.loc[:, "PV_panels"])
print("Generator Output Grid:", network.generators_t.p.loc[:, "Grid_gen"])
print("Load Consumption:", network.loads_t.p.loc[:, "Electric_load"])
print("SoC Battry:", network.stores_t.e["Battery"])
print("Power Battry:", network.stores_t.p["Battery"])
# Data for plotting
times = snapshots
load_data = network.loads_t.p.loc[:, "Electric_load"]
PV_panel_data = network.generators_t.p.loc[:, "PV_panels"]
Grid_data = network.generators_t.p.loc[:, "Grid_gen"]
Battery_data = network.stores_t.p["Battery"]
Battery_data_SoC = network.stores_t.e["Battery"]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(times, load_data, label='Load Consumption (kW)', marker='o')
plt.plot(times, PV_panel_data, label='Generator Output (kW)', marker='o')
plt.plot(times, Grid_data, label='Generator Output (kW)', marker='o')
plt.plot(times, Battery_data, label='Battery Output (kW)', marker='o')
plt.plot(times, Battery_data_SoC, label='Battery SoC (kWh)', marker='o')
plt.title('Power Generation vs Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.grid(True)
plt.legend()
plt.xticks(times, [time.strftime('%H:%M') for time in times])
plt.tight_layout()
plt.show()