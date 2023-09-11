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
end_date =  "2025-07-03 23:00"
storage_marginal_cost = 1
candi_hourly_match_portion = 0.10




#select the output graphs desired
stacked_gentech_w_batteries = True
stacked_gentech_batteries = True
stacked_demand_region = True
stacked_generators_nem = True
stacked_generators_region = True
batteries_nominal = False
batteries_normalised = False
regional_demand = False

#%%

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

#Read in the buses
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
            p_min_pu = row["p_min_pu"],
            carrier = row["carrier"],
            marginal_cost = row["marginal_cost"],
            start_up_cost = row["start_up_cost"]
            )
input_generators()

#%%

#add index and length of simulation
index = pd.date_range(start_date,end_date, freq="H")
network.set_snapshots(index)

#%%

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

#Input both the C&I loads and the remaining load
def input_loads():

    candi_loads = pd.read_csv("candi_hundred_percent.csv")

    #Scale specific columns of candi_loads to be 10%. 
    candi_loads.iloc[:, 1:11] *= candi_hourly_match_portion

    #Read in the loads
    loads = pd.read_csv("demand.csv")

    #Set the datetime column to be the index
    candi_loads.set_index('datetime', inplace=True)
    loads.set_index('datetime',inplace = True)

    #Create an empty DataFrame with the same index as loads and columns as candi_loads
    loads_less_candi = pd.DataFrame(index=loads.index, columns=candi_loads.columns)

    #Perform element-wise subtraction and fill the loads_less_candi DataFrame
    for column in loads_less_candi.columns:
        loads_less_candi[column] = loads[column] - candi_loads[column]

    #Now, lets input the two loads into the model   
    for regionID in candi_loads.columns[1:11]:
        temp_col = candi_loads[regionID].tolist()
        network.add("Load", "{} 24/7 C&I Demand".format(str(regionID)), bus=str(regionID), p_set=temp_col[startdate_hour:enddate_hour]) 
    for regionID in loads_less_candi.columns[1:11]:
        temp_col = loads_less_candi[regionID].tolist()
        network.add("Load", "{} NetDemand".format(str(regionID)), bus=str(regionID), p_set=temp_col[startdate_hour:enddate_hour])
    return candi_loads

candi_loads = input_loads()
 
#%%

#Input the links between each bus
def input_links():
    #Import the links
    links = pd.read_csv('links.csv', delimiter = ',')
    links["name"] = links["name"].astype(str)
    for index, row in links.iterrows(): 
            network.add(
                "Link",
                "{}".format(row["name"]),
                bus0 = row["bus0"],
                bus1 = row["bus1"],
                p_nom = row["p_nom"],
            )
input_links()


#%%

#Lets create a constraint on hydro that limits the outflow using a daily pro-rata rate
m = network.optimize.create_model()

max_hydro_inflow_per_day = 41980.45
max_hydro_inflow = max_hydro_inflow_per_day * no_of_days

gen_carriers = network.generators.carrier.to_xarray()

gen_p = m.variables["Generator-p"]

hydro_generators = gen_carriers.where(gen_carriers == "Hydro")

hydro_generation = gen_p.groupby(hydro_generators).sum().sum("snapshot")

constraint_expression1 = hydro_generation <= max_hydro_inflow
m.add_constraints(constraint_expression1, name="Hydro-Max_Generation")

#%%

#100% Annual Matching Constraint, this doesn't distinguish by bus yet

is_hydro = gen_carriers == "Hydro"
is_solar = gen_carriers == "Solar"
is_wind = gen_carriers == "Wind"
is_renewable = is_hydro | is_solar | is_wind

renewable_generators = gen_carriers.where(is_renewable)

#get a list of the buses
bus = network.generators.bus.to_xarray()

renewable_generation = gen_p.groupby(renewable_generators).sum().sum("snapshot")

candi_generation = candi_loads[startdate_hour:enddate_hour].sum().sum()

constraint_expression2 = renewable_generation >= candi_generation
m.add_constraints(constraint_expression2, name="100% RES")


annual_sourcing_constraint = renewable_generation >= C

#%%
#Now, lets optimise the model! Good luck champ
network.optimize.solve_model(solver_name = solver_name)

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
    if regional_demand == True:

        for region in network.loads_less_candi.index: 
            plt.plot(network.snapshots, network.loads_t.p[region])
            plt.ylabel('Load (MW)')
            plt.xlabel('Time')
            plt.title(region)
            plt.tight_layout()
            plt.show()

visualise()


print(network.statistics.curtailment(comps = None))

print("All done! Well done champion\n")