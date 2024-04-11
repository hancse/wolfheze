#!/usr/bin/env python
# coding: utf-8

# In[]:

import pypsa
import matplotlib.pyplot as plt
import pandas as pd


network = pypsa.Network()


# In[Buses]
network.add("Bus","AC_bus",carrier = 'AC')  #....Main AC_bus
network.add("Bus","40C_bus",carrier = 'Heat_40')#,carrier = "heat") #....40 degree celcius heat bus
network.add("Bus","60C_bus",carrier = 'Heat_60')#,carrier = "heat") #....60 degree celcius heat bus
network.add("Bus","H2_bus",carrier = 'H2')#,carrier = "gas")   #....Hydrogen Bus
network.add("Bus","48VDC_bus",carrier = 'DC')# 48V box inside PowerBox
network.add("Bus","House_bus", carrier = 'AC')# 8 houses bus

# In[Generators]


network.add("Generator",
            "PV_panels",
             bus="AC_bus",
             carrier ='AC',
             p_nom = 51.8,
             p_min_pu = 0,
             p_max_pu = 0.8,
             marginal_cost = 0.1)       #...Power from PV panels

# In[Generators]

network.add("Generator",
            "Grid_gen",
             bus="AC_bus",
             carrier ='AC',
             p_nom = 55.2,
             marginal_cost = 0.2) #power from grid bus

# In[Load 1]

network.add("Load",
            "Electric_load",
            bus="House_bus",
            p_set = 8) #Electric load connected to the house bus

# In[Load 2]

network.add("Load",
            "space_heating",
            bus ="40C_bus",
            carrier ='Heat_40',
            p_set = 40) #space heating to 40C for 8 houses, peak load

# In[Load 3]

network.add("Load",
            "EV_chargers",
            bus = "House_bus",
            p_set = 30) #1 charger per house 8 in total

# In[Load 4]

network.add("Load",
            "DHW_demand",
            bus ="60C_bus",
            carrier ='Heat_60',
            p_set = 2)   #DHW_demand added to 60C bus for 8houses

# In[Link 1]

network.add("Link",
            "Electrolyzer",
            bus0= "AC_bus",
            bus1= "H2_bus",
            p_nom = 5,
            efficiency=1) #Efficiency is not known yet, has to be checked in Datasheet!

# In[Link 2]

network.add("Link",
            "Inverter",
            bus0="48VDC_bus",
            bus1= "AC_bus",
            p_nom = 5,
            efficiency = 1) #Efficiency is not known.

# In[Link 3]

network.add("Link",
            "Feulcell",
            bus0= "H2_bus",
            bus1= "48VDC_bus",
            p_nom = 5,
            efficiency = 1) #Efficiency is not known.

# In[Link 4]

network.add("Link",
            "Heat pump",
            bus0="AC_bus",
            bus1="40C_bus",
            p_nom = 20,
            efficiency = 3)

# In[Link 5]

network.add("Link",
            "Booster Heat Pump",
            bus0="AC_bus",
            bus1="40C_bus",
            bus2="60C_bus",
            p_nom = 0.5,
            p_nom_max = 0.5,
            efficiency = -3,
            efficiency2 = 4)


# In[Link 6]

network.add("Link",
            "Booster Electric Element",
            bus0="AC_bus",
            bus1="60C_bus",
            p_nom = 2,
            efficiency = 1)

# In[Link 7]

network.add("Link",
            "GridFeedIn",
            bus0="Grid_gen",
            bus1="AC_bus",
            p_nom = 55.2,
            efficiency = -1,
            marginal_cost = 0.2)


# In[Link 8]

network.add("Link",
            "HouseConnection",
            bus0="AC_bus",
            bus1="House_bus",
            effiency = -1,
            p_nom = 138,
            p_nom_max= 138,
            )
# In[Store 1]

network.add("Store",
            "H2_tanks,",
            bus="H2_bus",
            e_nom_max = 1200,
            e_nom_min = 0,
            e_nom_opt = 10,
            p_nom = 5)

# In[Store 2]

#maybe a buffertank? can we assume this is not needed in the static model?
#network.add("Store",
#            "Buffertank",
#            bus= "40C_bus",
#            carrier = "Heat_40")

# In[Store 3]

network.add("Store",
            "Battery",
            bus= "48VDC_bus",
            e_nom_max = 30,
            e_nom_min = 0,
            p_nom = 5)

# In[]

network.optimize()

# In[]
network.lopf()

# In[]
#print(dir(network.links_t))

#network.generators_t
#network.stores_t
#network.buses_t
#network.links

network.links_t.p0, network.links_t.p1, network.links_t.p2

network.generators_t.p,network.loads_t.p

# In[]

pd.set_option('display.max_columns', None)
generator_power = network.generators_t.p["PV_panels"]

# Plot generator power output
plt.plot(generator_power.index, generator_power.values, label="PV_panels")
plt.xlabel("Time")
plt.ylabel("Power Output (kW)")
plt.title("Generator Power Output")
plt.legend()
plt.show()
