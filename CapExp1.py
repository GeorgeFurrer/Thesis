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
solver_name = "gurobi"  #Can be gurobi or GLPK
storage_marginal_cost = 1
candi_hourly_match_portion = 0.10 #portion of C&Is that hourly match
investment_years = [2030,2035,2040] #Years to do capacity expansion
optimise_frequency = 1 #hours per capacity expansion time slice
r = 0.025 #discount rate



#%%
#Set the snapshots to perform optimisation
def set_snapshots():
    snapshots = pd.DatetimeIndex([])
    for year in investment_years:
        period = pd.date_range(
            start="{}-01-01 00:00".format(year),
            freq="{}H".format(optimise_frequency),
            end = "{}-12-31 23:00".format(year)
        )
        snapshots = snapshots.append(period)


    # convert to multiindex and assign to network
    network.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    network.investment_periods = investment_years

    network.investment_period_weightings["years"] = list(np.diff(investment_years)) + [5]
set_snapshots()

#%%

#Set discount rate. Don't really know what this does honestly, I've modified it from PyPSA example
def set_discount_rate():
    T = 0

    for period, nyears in network.investment_period_weightings.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
        network.investment_period_weightings.at[period, "objective"] = sum(discounts)
        T += nyears
set_discount_rate()

#%%

#Read in the buses
def input_buses():
        
    #Import the buses
    buses = pd.read_csv('buses.csv', delimiter = ',')
    for index, row in buses.iterrows(): 
        network.add(
        "Bus",
        "{}".format(row["name"])
        )
input_buses()

#%%

#Input the carrier types and associated emissions
def input_carriers():
    carriers = pd.read_csv("carriers.csv")
    for index, row in carriers.iterrows():
        network.add(
            "Carrier", 
            "{}".format(row["Technology"]), 
            co2_emissions = row["Emissions_Intensity"]
        )
input_carriers()

#%%
#Read in all generators in 2025
def input_generators():
    generators = pd.read_csv('generators.csv',delimiter = ',')

    #ensure numerical data are floats and not strings

    generators['p_nom'] = generators['p_nom'].astype(float)
    generators['p_min_pu'] = generators['p_min_pu'].astype(float)
    generators['marginal_cost'] = generators['marginal_cost'].astype(float)
    generators['start_up_cost'] = generators['start_up_cost'].astype(float)

    #Add the Generators in
    for index,row in generators.iterrows():
        network.add(
        "Generator",
        "{}".format(row["name"]),
        bus = row["bus"],
        p_nom = row["p_nom"],
        p_min_pu = row["p_min_pu"],
        carrier = row["carrier"],
        marginal_cost = row["marginal_cost"],
        start_up_cost = row["start_up_cost"], 
        build_year = row["build_year"], 
        lifetime = row["lifetime"]
            )
input_generators()

#%%

#Input the variable renewables p_max_pu for all existings gens
def input_generators_t_p_max_pu():
    #input the generators_t.p_max_pu
    input_vre_trace = pd.read_csv("generators_t.p_max_pu_v02.csv", delimiter = ",")

    for column in input_vre_trace.columns[1:]:
        generator_name = column
        vre_trace_list = input_vre_trace[column].tolist()
        vre_trace_list_repeated = vre_trace_list*len(investment_years) + vre_trace_list[:24]
        #print(network.generators.loc[generator_name]) #copies trace data across each of the years (2030,2035,2040)
        network.generators_t.p_max_pu[generator_name] = vre_trace_list_repeated
input_generators_t_p_max_pu()

#%%

#Input the links between each bus (and future links)
def input_links():
    links = pd.read_csv('links_cap_exp.csv', delimiter = ',')
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

#Input the batteries that will exist in 2025
def input_batteries():
    #lets import some batteries 
    storage_units = pd.read_csv('storage_units.csv',delimiter = ',')
    storage_units["marginal_cost"] = storage_marginal_cost
    for index,row in storage_units.iterrows():
            network.add(
            "StorageUnit",
            "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            max_hours = row["max_hours"], 
            marginal_cost = row["marginal_cost"], 
            lifetime = row["lifetime"],
            capital_cost = row["capital_cost"],
            efficiency_store = row["efficiency_store"],
            efficiency_dispatch = row["efficiency_dispatch"]
            )
input_batteries()
        
#%%

def input_dummy_extendable_generators():
    dummy_gens = pd.read_csv("extendable_generators.csv")
    for index, row in dummy_gens.iterrows():
        network.add(
            "Generator",
             "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            p_min_pu = row["p_min_pu"],
            carrier = row["carrier"],
            marginal_cost = row["marginal_cost"],
            p_nom_extendable = True,
            build_year = row["build_year"],
            lifetime = row["lifetime"], 
            capital_cost = row["capital_cost"]
            )
input_dummy_extendable_generators()

#%%

def input_dummy_vre_trace_p_max_pu():
    dummy_vre_trace = pd.read_csv("VRE_Traces_By_Region_v02.csv")

    for column in dummy_vre_trace.columns[1:]:
            for investment_year in investment_years:
                dummy_gen_name = column + " " + str(investment_year)
                dummy_vre_trace_list = dummy_vre_trace[column].tolist()
                #print(network.generators.loc[dummy_gen_name]) #copies trace data across each of the years (2030,2035,2040)
                network.generators_t.p_max_pu[dummy_gen_name] = dummy_vre_trace_list
input_dummy_vre_trace_p_max_pu()

#%%

#Input Extendable/Dummy Batteries 
def input_dummy_extendable_batteries(): 
    dummy_batteries = pd.read_csv("extendable_storage_units.csv")
    #set marginal storage cost to user input (usually 1)
    dummy_batteries["marginal_cost"] = storage_marginal_cost
    for index, row in dummy_batteries.iterrows(): 
            network.add(
            "StorageUnit",
            "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            max_hours = row["max_hours"],
            marginal_cost = row["marginal_cost"],
            p_nom_extendable = True,
            build_year = row["build_year"],
            lifetime = row["lifetime"],
            capital_cost = row["capital_cost"], 
            efficiency_store = row["efficiency_store"],
            efficiency_dispatch = row["efficiency_dispatch"]
            )
input_dummy_extendable_batteries()

#%%

#Input both the C&I loads and the remaining load
def input_loads():

    #Read in C&I matching loads and total load file
    candi_matching_loads = pd.read_csv("candi_demand_cap_exp_v01.csv")
    loads = pd.read_csv("demand_cap_exp_v01.csv")

    #Scale specific columns of candi_matching_loads 
    candi_matching_loads.iloc[:, 1:] *= candi_hourly_match_portion
    
    candi_matching_loads.set_index('Date', inplace=True)
    loads.set_index('Index',inplace = True)

    #Create df for the total load, net the C&I matching component
    loads_less_candi_matching = pd.DataFrame(index=loads.index, columns=candi_matching_loads.columns)

    #Perform element-wise subtraction and fill the loads_less_candi_matching DataFrame
    for column in loads_less_candi_matching.columns:
        loads_less_candi_matching[column] = loads[column] - candi_matching_loads[column]

    #Now, lets input the two loads into the model   
    for regionID in candi_matching_loads.columns[1:11]:
        temp_col = candi_matching_loads[regionID].tolist()
        network.add("Load", "{} 24/7 C&I Demand".format(str(regionID)), bus=str(regionID), p_set=temp_col) 
    for regionID in loads_less_candi_matching.columns[1:11]:
        temp_col = loads_less_candi_matching[regionID].tolist()
        network.add("Load", "{} NetDemand".format(str(regionID)), bus=str(regionID), p_set=temp_col)
    return candi_matching_loads
candi_matching_loads = input_loads()

#%%

#This constraint puts a maximum cap on the hydro allowed to be dispatched
def hydro_constraint():
    #Lets create a constraint on hydro that limits the outflow using a daily pro-rata rate
    m = network.optimize.create_model()

    max_hydro_inflow_per_day = 41980.45
    max_hydro_inflow = max_hydro_inflow_per_day * (365*len(investment_years) + 1) #2040 is a leap year

    gen_carriers = network.generators.carrier.to_xarray() #creates an array of the carrier of all gens

    gen_p = m.variables["Generator-p"] 

    hydro_generators = gen_carriers.where(gen_carriers == "Hydro")

    hydro_generation = gen_p.groupby(hydro_generators).sum().sum("snapshot")

    constraint_expression1 = hydro_generation <= max_hydro_inflow
    m.add_constraints(constraint_expression1, name="Hydro-Max_Generation")
hydro_constraint()

#%%

def annual_matching(): 
    #100% Annual Matching Constraint, this doesn't distinguish by bus yet

    is_hydro = gen_carriers == "Hydro"
    is_solar = gen_carriers == "Solar"
    is_wind = gen_carriers == "Wind"
    is_renewable = is_hydro | is_solar | is_wind

    renewable_generators = gen_carriers.where(is_renewable)

    #get a list of the buses
    bus = network.generators.bus.to_xarray()

    renewable_generation = gen_p.groupby(renewable_generators).sum().sum("snapshot")

    candi_generation = candi_matching_loads.sum().sum()
    #can add [startdatehour:enddatehour] before sum.().sum()
    constraint_expression2 = renewable_generation >= candi_generation
    m.add_constraints(constraint_expression2, name="100% RES")

#%%

network.optimize.solve_model(multi_investment_periods=True, solver_name = solver_name)
print("All done! Well done champion\n")

# %%


#JUNK


def visualise(): 
    #Plot capacity in each time slice by technology
    colours = ["saddlebrown","black","red", "orange", "blue", "yellow", "green"]

    d = ["StorageUnit","Generator"]
    for c in d:
        df = pd.concat(
            {
                period: network.get_active_assets(c, period) * network.df(c).p_nom_opt
                for period in network.investment_periods
            },
            axis=1,
        )
        df.T.plot.bar(
            stacked=True,
            edgecolor="white",
            width=1,
            ylabel="Capacity",
            xlabel="Investment Period",
            rot=0,
            figsize=(10, 5),
        )
        plt.tight_layout()
        plt.show()


    for c in d:
        df = pd.concat(
            {
                period: network.get_active_assets(c, period) * network.df(c).p_nom_opt
                for period in network.investment_periods
            },
            axis=1,
        )
        df.T.groupby(network.df("Generator").carrier).plot.bar(
            stacked=True,
            edgecolor="white",
            width=1,
            ylabel="Capacity",
            xlabel="Investment Period",
            rot=0,
            figsize=(10, 5),
            color = colours
        )
        plt.tight_layout()
        plt.show()

visualise()

# #OUTPUT
# def visualise():
#     if stacked_gentech_batteries == True:
#         # Group generators by carrier and sum their generation and plot the stacked area graph
#         colours = ["black","brown", "orange", "blue", "yellow", "green"]
#         grouped_generators = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum()
#         ax1 = grouped_generators.plot.area(ylabel="Generation (MW)", xlabel="Time", zorder=True, color = colours)
#         plt.legend(bbox_to_anchor=(1.1, 1), fontsize = 6)

#         ax2 = ax1.twinx()
#         ax2 = network.storage_units_t.state_of_charge.plot.line(ylabel = "Energy (MWh)", ax=ax2,zorder = True)
#         ax2.set_ylabel("Energy (MWh)")
#         plt.tight_layout()
#         plt.legend(bbox_to_anchor=(2, 1), fontsize = 6) 
#         plt.show()

#     if stacked_demand_region == True:
#         #Plot the load 
#         network.loads_t.p.plot.area(ylabel = "Demand (MW)")
#         plt.legend(bbox_to_anchor=(1, 1), fontsize = 6)
#         plt.tight_layout()
#         plt.show()
#     if stacked_generators_nem == True:
#         network.generators_t.p.plot.area(ylabel = "Demand (MW)",xlabel = "Time", zorder = True)
#         plt.legend(fontsize=6, bbox_to_anchor = (1,1))  
#         plt.tight_layout()
#         plt.show()
#     if batteries_nominal == True:
#         #plot storage units state of charge (nominal values)
#         network.storage_units_t.state_of_charge.plot.line(ylabel = "State of Charge", xlabel = "Time", zorder = True)
#         plt.legend(bbox_to_anchor=(1,1), fontsize = 6)
#         plt.tight_layout()
#         plt.show()
#     if batteries_normalised == True:
#         #plot storage units normalised state of charge 
#         for storage_unit in network.storage_units.index:
#             state_of_charge = network.storage_units_t.state_of_charge[storage_unit]
#             max_state_of_charge = network.storage_units.at[storage_unit, 'max_hours'] * network.storage_units.at[storage_unit, 'p_nom']

#             # Calculate the normalized state of charge as a percentage
#             normalized_state_of_charge = (state_of_charge / max_state_of_charge) * 100

#             # Plot the normalized state of charge for each battery
#             plt.plot(network.snapshots, normalized_state_of_charge, label=storage_unit)

#         plt.xlabel('Time')
#         plt.ylabel('State of Charge (% of max)')
#         plt.legend(fontsize=6, bbox_to_anchor = (1,1))
#         plt.tight_layout()
#         plt.show()
#     if regional_demand == True:

#         for region in network.loads_less_candi_matching.index: 
#             plt.plot(network.snapshots, network.loads_t.p[region])
#             plt.ylabel('Load (MW)')
#             plt.xlabel('Time')
#             plt.title(region)
#             plt.tight_layout()
#             plt.show()

# visualise()

# %%
