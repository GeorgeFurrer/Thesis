#%%
import pypsa 
import pandas as pd
import matplotlib.pyplot as plt
import logging 
import numpy as np
from datetime import datetime
import xarray as xr

logging.basicConfig(level=logging.INFO)
#%%
#Lets create a network
network = pypsa.Network()

#INPUTS - Fill in as desired
solver_name = "glpk"
start_date = "2025-07-01 00:00"
end_date =  "2025-07-01  23:00"
storage_marginal_cost = 1


#select the output graphs desired
stacked_gentech_w_batteries = True
stacked_gentech_batteries = True
stacked_demand_region = True
stacked_generators_nem = True
stacked_generators_region = False
batteries_nominal = True
batteries_normalised = True
regional_demand = True

#function determines the hour number of the year, as well as the number of days in the optimise period

def hour_number_of_year(start_date,end_date):

    stdt = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    enddt = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

    start_of_year = datetime(stdt.year, 1, 1)
    start_hour_number = (stdt - start_of_year).total_seconds() // 3600
    end_hour_number = (enddt - start_of_year).total_seconds() // 3600 + 1

    hour_difference = end_hour_number - start_hour_number
    no_of_days = (hour_difference) / 24 

    return int(start_hour_number), int(end_hour_number), float(no_of_days)
startdate_hour, enddate_hour, no_of_days = hour_number_of_year(start_date, end_date)


#%%

#read in the buses
def input_buses():
        
    #Import the buses
    buses = pd.read_csv('buses.csv', delimiter = ',')
    for index, row in buses.iterrows(): 
        network.add(
        "Bus",
        "{}".format(row["name"]),
        #v_nom = row["v_nom"]
        )
input_buses()


#%%

#read in all generators
def input_generators():
    generators = pd.read_csv('generators.csv',delimiter = ',')

    #ensure numerical data are floats and not strings
    generators['p_nom'] = generators['p_nom'].astype(float)
    generators['p_min_pu'] = generators['p_min_pu'].astype(float)
    generators['marginal_cost'] = generators['marginal_cost'].astype(float)
    generators['start_up_cost'] = generators['start_up_cost'].astype(float)

    #Add the Generators in
    for index,row in generators.iterrows():
        #check asset hasn't come offline
        if row["expected_retirement_year"] >= 2025:
            network.add(
            "Generator",
            "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            #p_min_pu = row["p_min_pu"],
            carrier = row["carrier"],
            marginal_cost = row["marginal_cost"],
            start_up_cost = row["start_up_cost"]
            )
input_generators()


#%%

#add index and length of simulation
index = pd.date_range(start_date,end_date, freq="H")
network.set_snapshots(index)

#Input the variable renewables p_max_pu
def input_generators_t_p_max_pu():
    #input the generators_t.p_max_pu
    input_vre_trace = pd.read_csv("generators_t.p_max_pu.csv", delimiter = ",")

    for column in input_vre_trace.columns[1:]:
        generator_name = column
        vre_trace_list = input_vre_trace[column][startdate_hour:enddate_hour].tolist()
        network.generators_t.p_max_pu[generator_name] = vre_trace_list
input_generators_t_p_max_pu()

#%%

#Input the batteries
def input_batteries():
    #lets import some batteries 
    storage_units = pd.read_csv('storage_units.csv',delimiter = ',')
    storage_units["marginal_cost"] = storage_marginal_cost
    for index,row in storage_units.iterrows():
        #if row["bus"] == "VIC": #or row["bus"] == "TAS":
            network.add(
            "StorageUnit",
            "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            max_hours = row["max_hours"], 
            marginal_cost = row["marginal_cost"]
            )
input_batteries()
        
#%%

def input_loads():
    loads = pd.read_csv("demand.csv", index_col=0)
    for regionID in loads.columns[1:11]:
        #if loads[regionID].name == "VIC":
            temp_load = loads[regionID].tolist()
            network.add("Load", "{} demand".format(str(loads[regionID].name)), bus=str(loads[regionID].name), p_set=temp_load[startdate_hour:enddate_hour])
input_loads()

#Input the links between each bus
def input_links():
    #Import the links
    links = pd.read_csv('links.csv', delimiter = ',')
    links["name"] = links["name"].astype(str)
    for index, row in links.iterrows(): 
        #if row["name"] == "TAS - VIC" or  row["name"] == "VIC - TAS":
            network.add(
                "Link",
                "{}".format(row["name"]),
                bus0 = row["bus0"],
                bus1 = row["bus1"],
                p_nom = row["p_nom"],
            )
input_links()


#%%

#Now, lets optimise the grid
network.optimize(network.snapshots, solver_name = solver_name)


#%%

# #Lets create a constraint on hydro that limits the outflow using a daily pro-rata rate
# m = network.optimize.create_model()

# max_hydro_inflow_per_day = 41980.45
# max_hydro_inflow = max_hydro_inflow_per_day * no_of_days

# gen_carriers = network.generators.carrier.to_xarray()

# #Returns array of all hydro gens with their specs
# hydro_generators = gen_carriers.where(gen_carriers == "Hydro", drop = True)

# gen_p = m.variables["Generator-p"]

# hydro_generation = gen_p.groupby(hydro_generators).sum().sum("snapshot")

# constraint_expression = hydro_generation <= max_hydro_inflow
# m.add_constraints(constraint_expression, name="Hydro-Max_Generation")

# network.optimize.solve_model()


#%%



#OUTPUT

def visualise():
    if stacked_gentech_batteries == True:
        # Group generators by carrier and sum their generation and plot the stacked area graph
        colours = ["black","brown", "orange", "blue", "yellow", "green"]
        grouped_generators = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum()
        ax1 = grouped_generators.plot.area(ylabel="Generation (MW)", xlabel="Time", zorder=True, color = colours)
        plt.legend(bbox_to_anchor=(1.1, 1), fontsize = 6)

        ax2 = ax1.twinx()
        ax2 = network.storage_units_t.state_of_charge.plot.line(ylabel = "Energy (MWh)", ax=ax2,zorder = True)
        ax2.set_ylabel("Energy (MWh)")
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(2, 1), fontsize = 6) 
        plt.show()
    if stacked_demand_region == True:
        #Plot the load 
        network.loads_t.p.plot.area(ylabel = "Demand (MW)")
        plt.legend(bbox_to_anchor=(1, 1), fontsize = 6)
        plt.tight_layout()
        plt.show()
    if stacked_generators_nem == True:
        network.generators_t.p.plot.area(ylabel = "Demand (MW)",xlabel = "Time", zorder = True)
        plt.legend(fontsize=6, bbox_to_anchor = (1,1))  
        plt.savefig('all_gens.png', bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    if batteries_nominal == True:
        #plot storage units state of charge (nominal values)
        network.storage_units_t.state_of_charge.plot.line(ylabel = "State of Charge", xlabel = "Time", zorder = True)
        plt.legend(bbox_to_anchor=(1,1), fontsize = 6)  
        plt.tight_layout()
        plt.show()

    if batteries_normalised == True:
        #plot storage units normalised state of charge 
        for storage_unit in network.storage_units.index:
            state_of_charge = network.storage_units_t.state_of_charge[storage_unit]
            max_state_of_charge = network.storage_units.at[storage_unit, 'max_hours'] * network.storage_units.at[storage_unit, 'p_nom']

            # Calculate the normalized state of charge as a percentage
            normalized_state_of_charge = (state_of_charge / max_state_of_charge) * 100

            # Plot the normalized state of charge for each battery
            plt.plot(network.snapshots, normalized_state_of_charge, label=storage_unit)

        plt.xlabel('Time')
        plt.ylabel('State of Charge (% of max)')
        plt.legend(fontsize=6, bbox_to_anchor = (1,1))
        plt.tight_layout()
        plt.show()

visualise()

print(network.statistics.curtailment(comps = None))

#%%
    #plot each region output 
    for region in network.loads.index: 
        plt.plot(network.snapshots, network.loads_t.p[region])
        plt.ylabel('Load (MW)')
        plt.xlabel('Time')
        plt.title(region)
        plt.tight_layout()
        plt.show()


#%%
    # if stacked_generators_region == True: 
    #     regions = network.generators['bus'].unique()
    #     for region in regions:
    #         plt.plot(network.snapshots, network.generators_t.p.groupby(network.generators.carrier,axis=1).sum())
    #         plt.legend(fontsize=6, bbox_to_anchor = (1,1))
    #         plt.show()


# %%

print("All done!")
# %%


