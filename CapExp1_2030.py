#%%

import pypsa 
import pandas as pd
import matplotlib.pyplot as plt
import logging 

import numpy as np
from datetime import datetime
import xarray as xr
import sys

logging.basicConfig(level=logging.INFO)

#%%
#Lets create a network
network = pypsa.Network()

#INPUTS - Fill in as desired
solver_name = "gurobi"  #Can be gurobi or GLPK
candi_hourly_match_portion = 0.10
 #portion of C&Is that hourly match
investment_years = [2030,2035,2040] #Years to do capacity_output expansion
optimise_frequency = 1 #hours per capacity expansion time slice
r = 0.055 #discount rate
upscale_demand_factor = 1 
rpp = 0.1896
CFE_score = 1
folder_path = 'Results_csvs'
#hydro_RE = False



#%%
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
#Set the snapshots to perform optimisation

#%%

def set_discount_rate():
    T = 0

    for period, nyears in network.investment_period_weightings.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
        network.investment_period_weightings.at[period, "objective"] = sum(discounts)
        T += nyears
set_discount_rate()
#Set discount rate. 
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

def input_carriers():
    carriers = pd.read_csv("carriers.csv")
    for index, row in carriers.iterrows():
        network.add(
            "Carrier",
            "{}".format(row["Technology"]),
            co2_emissions = row["Emissions_Intensity"]
        )
input_carriers()
#Input the carrier types and associated emissions

#%%

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
#Input the links between each bus (and future links)

#%%

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
        #p_min_pu = row["p_min_pu"],
        carrier = row["carrier"],
        marginal_cost = row["marginal_cost"],
        start_up_cost = row["start_up_cost"], 
        build_year = row["build_year"], 
        lifetime = row["lifetime"]
            )
input_generators()
#Read in all generators in 2025

#%%

def input_generators_t_p_max_pu():
    
    # Read the data and convert it to a NumPy array
    input_vre_trace = pd.read_csv("generators_t.p_max_pu_v03.csv", delimiter=",")

    # Extract the column names
    column_names = input_vre_trace.columns[1:]

    # Convert the data to a NumPy array
    input_vre_trace = input_vre_trace.iloc[:, 1:].to_numpy()

    for column_idx in range(input_vre_trace.shape[1]):
        generator_name = column_names[column_idx]
        vre_trace_list = input_vre_trace[:, column_idx]
        # This clips the values between 0.001 and 0 to 0
        vre_trace_list = np.clip(vre_trace_list, 0.001, None)
        vre_trace_list = vre_trace_list.astype(np.float16)
        repeated_trace = np.tile(vre_trace_list, len(investment_years))
        if len(network.investment_periods) == 3:
            repeated_trace = np.concatenate([repeated_trace, vre_trace_list[:24]])
        
        network.generators_t.p_max_pu[generator_name] = repeated_trace
input_generators_t_p_max_pu()
#Input the variable renewables p_max_pu for all existings gens

#%%


def input_batteries():
    #lets import some batteries 
    storage_units = pd.read_csv('storage_units.csv',delimiter = ',')
    for index,row in storage_units.iterrows():
            network.add(
            "StorageUnit",
            "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            max_hours = row["max_hours"], 
            marginal_cost = row["marginal_cost"], 
            lifetime = row["lifetime"],
            #capital_cost = row["capital_cost"].astype("float16"),
            efficiency_store = row["efficiency_store"],
            efficiency_dispatch = row["efficiency_dispatch"]
            )
input_batteries()
#Input the batteries that will exist in 2025     
#%%

"""CAPACITY EXPANSION SECTION """


def input_dummy_extendable_generators():
    dummy_gens = pd.read_csv("extendable_generators.csv")

    for index, row in dummy_gens.iterrows():
        network.add(
            "Generator",
             "{}".format(row["name"]),
            bus = row["bus"],
            p_nom = row["p_nom"],
            #p_min_pu = row["p_min_pu"],
            carrier = row["carrier"],
            marginal_cost = row["marginal_cost"],
            p_nom_extendable = True,
            build_year = row["build_year"],
            lifetime = int(row["lifetime"]), 
            capital_cost = row["capital_cost"]
        )
input_dummy_extendable_generators()

#%%

def input_dummy_vre_trace_p_max_pu():
    dummy_vre_trace = pd.read_csv("VRE_Traces_By_Region_v03.csv")

    for column in dummy_vre_trace.columns[1:]:
            for investment_year in investment_years:
                dummy_gen_name = column + " " + str(investment_year)
                dummy_vre_trace_list = dummy_vre_trace[column][:len(network.snapshots)].tolist()
                dummy_vre_trace_list = np.array(dummy_vre_trace_list, dtype=np.float16)
                #print(network.generators.loc[dummy_gen_name]) #copies trace data across each of the years (2030,2035,2040)
                network.generators_t.p_max_pu[dummy_gen_name] = dummy_vre_trace_list
input_dummy_vre_trace_p_max_pu()

#%%

def input_dummy_extendable_batteries(): 
    dummy_batteries = pd.read_csv("extendable_storage_units.csv")
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
#Input Extendable/Dummy Batteries\

#%%

def input_loads():

    #Read in C&I matching loads and total load file
    candi_matching_loads = pd.read_csv("candi_demand_cap_exp_v01.csv")
    loads = pd.read_csv("demand_cap_exp_v01.csv")
    #loads.iloc[:, 1:] = loads.iloc[:, 1:].clip(lower=0)

    loads.iloc[:,1:] *= upscale_demand_factor

    #Scale specific columns of candi_matching_loads 
    candi_matching_loads.iloc[:, 1:] *= candi_hourly_match_portion
    
    candi_matching_loads.set_index('Date', inplace=True)
    loads.set_index('Index',inplace = True)

    #Create df for the total load, net the C&I matching component
    loads_less_candi_matching = pd.DataFrame(index=loads.index, columns=candi_matching_loads.columns)

    #Perform element-wise subtraction and fill the loads_less_candi_matching DataFrame
    for column in loads_less_candi_matching.columns:
        loads_less_candi_matching[column] = loads[column] - candi_matching_loads[column]
        loads_less_candi_matching[column] = loads_less_candi_matching[column].clip(lower=0)
    #Now, lets input the two loads into the model   
    for regionID in candi_matching_loads.columns[:11]:
        temp_col = candi_matching_loads[regionID][:len(network.snapshots)].tolist()
        temp_col = np.array(temp_col,dtype=np.float16)
        network.add("Load", "{} 24/7 C&I Demand".format(str(regionID)), bus=str(regionID), p_set=temp_col) 
    for regionID in loads_less_candi_matching.columns[:11]:
        temp_col = loads_less_candi_matching[regionID][:len(network.snapshots)].tolist()
        temp_col = np.array(temp_col,dtype=np.float16)
        network.add("Load", "{} NetDemand".format(str(regionID)), bus=str(regionID), p_set=temp_col)
    return candi_matching_loads, loads_less_candi_matching
candi_matching_loads,loads_less_candi_matching = input_loads()
#Input both the C&I loads and the remaining load

#%%

def load_profile():
    for period in network.investment_periods:
        hrno = int((period - 2030)/5*8760) + 48
        load_profile = network.loads_t.p_set[hrno:(hrno+48)]
        ax = load_profile.plot.area()
        plt.legend(bbox_to_anchor = (1,1))
        plt.ylabel("Generation (MW)")
        x_ticks = range(0, len(load_profile), 6)
        x_labels = load_profile.index.get_level_values(1)[x_ticks].strftime('%m-%d %H:%M')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation =90)
        plt.title(period)
        plt.show()
load_profile()

#%%
"""CONSTRAINTS"""

m = network.optimize.create_model(multi_investment_periods = True)

#%%

def hydro_constraint():
    max_hydro_inflow_per_day = 41980.45
    max_hydro_inflow = (max_hydro_inflow_per_day*(len(network.snapshots)/24))

    gen_carriers = network.generators.carrier
    gen_carriers = gen_carriers.to_xarray()
    gen_p = m.variables["Generator-p"] 

    hydro_generators = gen_carriers.where(gen_carriers == "Hydro")

    hydro_generation = gen_p.groupby(hydro_generators).sum().sum("snapshot")

    constraint_expression1 = hydro_generation <= max_hydro_inflow
    m.add_constraints(constraint_expression1, name="Hydro-Max_Generation")
hydro_constraint()

#%%

def annual_regional_matching(): # Constraint ensures RE generation in a region over the year > C&I consumption over the year in the same region
    
    for period in network.investment_periods:

        #RHS - Needs to be a fixed value. Energy Consumed by CandI loads, plus the RPP 

        period_start = int(((period - 2030)/5)*8760)
        period_end  = int((((period - 2030)/5)+ 1)*8760)

        matching_candi_consumption = pd.DataFrame()

        matching_candi_consumption["bus"] = network.loads.bus[:10] #first 10 cols are the C&I matching loads
        matching_candi_consumption["Matching_Consumption_(MWh)"] = network.loads_t.p_set.iloc[period_start:period_end, :10].sum() #2nd 10 loads are the net loads (that don't match)

        non_matching_rpp_consumption = pd.DataFrame(network.loads_t.p_set.iloc[period_start:period_end, 10:].sum()) * rpp

        #Need to reset indexes so we can add one column to the matching_candi_consumption dataframe
        matching_candi_consumption = matching_candi_consumption.reset_index()
        non_matching_rpp_consumption = non_matching_rpp_consumption.reset_index()

        matching_candi_consumption['non_matching_rpp_consumption'] = non_matching_rpp_consumption.iloc[:, 1]
        matching_candi_consumption = matching_candi_consumption.set_index('bus')

        matching_candi_consumption["Net_RE_Consumption"] = matching_candi_consumption["Matching_Consumption_(MWh)"] + matching_candi_consumption["non_matching_rpp_consumption"]
        
        for region in network.buses.index:
            #LHS - Variables. Renewable Energy Generation, by region across the year

            #First, return an x_array of the carrier/technology of each asset 
            gen_carriers = network.generators.carrier*network.get_active_assets("Generator",period)
            gen_carriers = gen_carriers.to_xarray()

            is_hydro = gen_carriers == "Hydro"
            is_solar = gen_carriers == "Solar"
            is_wind = gen_carriers == "Wind"
            is_renewable = is_hydro | is_solar | is_wind
            gen_p = m.variables["Generator-p"]
            renewable_power_variables = gen_p.where(is_renewable)
            # double checking using pandas
            # code below converts to pandas DataFrame, checks where the value is -1
            # linopy fills locations where the condition is false with -1, so the list printed
            # should only have renewable energy generators
            test = renewable_power_variables.to_pandas()
            test_2030 = test.loc[(period,)]
            test_2030.where(test_2030 > 0).dropna(axis=1).columns.tolist()
            # groupby buses
            gen_buses = network.generators.bus.to_xarray()
            #New COde
            curr_gen_bus = gen_buses == region
            bus_variables = gen_buses.where(curr_gen_bus)
            # This line below gives you the sum of all renewable energy generators at each bus and for each hour
            # So this should give you the ability to do regional hourly matching
            hourly_renewable_power_bus_sum = renewable_power_variables.groupby(bus_variables).sum() #bus_variables used to be gen_buses
            # Add `.sum("snapshot")` to produce annual LHS for each bus
            annual_renewable_bus_power_sum = renewable_power_variables.loc[period].groupby(bus_variables).sum("Generator").sum()
            net_re_consumption_by_region = matching_candi_consumption.loc[region]["Net_RE_Consumption"]

            constraint_expression_ann_match = annual_renewable_bus_power_sum >= net_re_consumption_by_region
            #print(constraint_expression_ann_match)
            m.add_constraints(constraint_expression_ann_match, name="100% RES_{}_{}".format(region,period))

#annual_regional_matching()

#%%

def annual_matching_by_NEM(): #Constraint ensures RE Generation across the NEM for the year > C&I demand in the NEM for the year
            #matching_candi_consumption["non_candi_RPP_consumption"] = matching_candi_consumption["non_candi_RPP_consumpti    #100% Annual Matching Constraint, this doesn't distinguish by bus yet
    for period in network.investment_periods:

        #RHS - Needs to be a fixed value. Energy Consumed by CandI loads, plus the RPP 

        period_start = int(((period - 2030)/5)*8760)
        period_end  = int((((period - 2030)/5)+ 1)*8760)

        matching_candi_consumption = pd.DataFrame()

        matching_candi_consumption["bus"] = network.loads.bus[:10] #first 10 cols are the C&I matching loads
        matching_candi_consumption["Matching_Consumption_(MWh)"] = network.loads_t.p_set.iloc[period_start:period_end, :10].sum() #2nd 10 loads are the net loads (that don't match)

        non_matching_rpp_consumption = pd.DataFrame(network.loads_t.p_set.iloc[period_start:period_end, 10:].sum()) * rpp

        #Need to reset indexes so we can add one column to the matching_candi_consumption dataframe
        matching_candi_consumption = matching_candi_consumption.reset_index()
        non_matching_rpp_consumption = non_matching_rpp_consumption.reset_index()

        matching_candi_consumption['non_matching_rpp_consumption'] = non_matching_rpp_consumption.iloc[:, 1]
        matching_candi_consumption = matching_candi_consumption.set_index('bus')

        matching_candi_consumption["Net_RE_Consumption"] = matching_candi_consumption["Matching_Consumption_(MWh)"] + matching_candi_consumption["non_matching_rpp_consumption"]
        
        #LHS - Variables. Renewable Energy Generation, by region across the year

        #First, return an x_array of the carrier/technology of each asset 
        gen_carriers = network.generators.carrier*network.get_active_assets("Generator",period)
        gen_carriers = gen_carriers.to_xarray()

        is_hydro = gen_carriers == "Hydro"
        is_solar = gen_carriers == "Solar"
        is_wind = gen_carriers == "Wind"
        is_renewable = is_hydro | is_solar | is_wind
        gen_p = m.variables["Generator-p"]
        renewable_power_variables = gen_p.where(is_renewable)

        annual_renewable_gen_sum = renewable_power_variables.sum("Generator").loc[period].sum() #Doesn't really work
        net_re_consumption = matching_candi_consumption["Net_RE_Consumption"].sum()

        constraint_expression_ann_match = annual_renewable_gen_sum >= net_re_consumption

        m.add_constraints(constraint_expression_ann_match, name="100% RES_{}".format(period))

annual_matching_by_NEM()


#%%

def hourly_matching_by_NEM(): #Ensures RE generation in the NEM exceeds aggregate C&I demand in every hour
    i = 0
    #RHS - Need to get a DF where each row represents a bus, and each 
    hourly_match_candi_demand = pd.DataFrame()
    hourly_match_candi_demand = network.loads_t.p_set.iloc[:,:10] * CFE_score
    #Rename column names to be the same as the buses
    hourly_match_candi_demand.columns = [col.split()[0] for col in hourly_match_candi_demand.columns]
    hourly_match_candi_demand["NEM"] = hourly_match_candi_demand.iloc[:,:10].sum(axis =1)

    #LHS - Variables are Generation from RE assets, + dispat+ch - storage of batteries
    for snapshot in network.snapshots:

        #for region in network.buses.index:
        #FIRST - RE Generation variable set up 
        gen_carriers = network.generators.carrier*network.get_active_assets("Generator",snapshot[0])
        gen_carriers = gen_carriers.to_xarray()

        is_hydro = gen_carriers == "Hydro"
        is_solar = gen_carriers == "Solar"
        is_wind = gen_carriers == "Wind"

        is_renewable = is_solar | is_wind | is_hydro 
        gen_p = m.variables["Generator-p"]
        renewable_power_variables = gen_p.where(is_renewable)
        gen_buses = network.generators.bus.to_xarray()
        #curr_gen_bus = gen_buses == region
        #bus_variables = gen_buses.where(curr_gen_bus)

        hourly_renewable_power_bus_sum = renewable_power_variables.sum("Generator")
        hourly_renewable_power_bus_sum = hourly_renewable_power_bus_sum.loc[snapshot[0]].loc[snapshot[1]]

        #SECOND - Battery dispatch variable set up 
        battery_dispatch = m.variables["StorageUnit-p_dispatch"]
        battery_store = m.variables["StorageUnit-p_store"]

        #battery_buses = network.storage_units.bus.to_xarray()
        #curr_batt_bus = battery_buses == region
        #battery_buses = battery_buses.where(curr_batt_bus)
        
        hourly_battery_dispatch_bus_sum = battery_dispatch.sum("StorageUnit")
        hourly_battery_dispatch_bus_sum = hourly_battery_dispatch_bus_sum.loc[snapshot[0]].loc[snapshot[1]]
        hourly_battery_store_bus_sum = battery_store.sum("StorageUnit")
        hourly_battery_store_bus_sum =hourly_battery_store_bus_sum.loc[snapshot[0]].loc[snapshot[1]]

        LHS_Variable = hourly_renewable_power_bus_sum + hourly_battery_dispatch_bus_sum - hourly_battery_store_bus_sum
        

        #Now to add this as a constraint into the model 
        constraint_expressions_hourly_match = LHS_Variable >= hourly_match_candi_demand.loc[snapshot]["NEM"]
        m.add_constraints(constraint_expressions_hourly_match, name="Hourly_Match_{}".format(snapshot))
        i = i + 1
        print(i)
hourly_matching_by_NEM()

#%%

def hourly_matching_by_region(): #NOT FINISHED - Ensures RE generation in every region exceeds C&I demand in every region in every hour. 
    i = 0
    #RHS - Need to get a DF where each row represents a bus, and each 
    hourly_match_candi_demand = pd.DataFrame()
    hourly_match_candi_demand = network.loads_t.p_set.iloc[:,:10] * CFE_score
    #Rename column names to be the same as the buses
    hourly_match_candi_demand.columns = [col.split()[0] for col in hourly_match_candi_demand.columns]

    #LHS - Variables are Generation from RE assets, + dispatch - storage of batteries
    for snapshot in network.snapshots[:2]:

        for region in network.buses.index:
            #FIRST - RE Generation variable set up 
            gen_carriers = network.generators.carrier*network.get_active_assets("Generator",snapshot[0])
            gen_carriers = gen_carriers.to_xarray()

            is_hydro = gen_carriers == "Hydro"
            is_solar = gen_carriers == "Solar"
            is_wind = gen_carriers == "Wind"

            is_renewable = is_hydro | is_solar | is_wind
            gen_p = m.variables["Generator-p"]
            renewable_power_variables = gen_p.where(is_renewable)
            gen_buses = network.generators.bus.to_xarray()
            curr_gen_bus = gen_buses == region
            bus_variables = gen_buses.where(curr_gen_bus)

            hourly_renewable_power_bus_sum = renewable_power_variables.loc[snapshot[0]].loc[snapshot[1]].groupby(bus_variables)
            hourly_renewable_power_bus_sum = hourly_renewable_power_bus_sum.sum("Generator")
            #SECOND - Battery dispatch variable set up 
            battery_dispatch = m.variables["StorageUnit-p_dispatch"]
            battery_store = m.variables["StorageUnit-p_store"]

            battery_buses = network.storage_units.bus.to_xarray()
            curr_batt_bus = battery_buses == region
            battery_buses = battery_buses.where(curr_batt_bus)

            hourly_battery_dispatch_bus_sum = battery_dispatch.loc[snapshot[0]].loc[snapshot[1]].groupby(battery_buses)
            hourly_battery_dispatch_bus_sum = hourly_battery_dispatch_bus_sum.sum("StorageUnit")
            hourly_battery_store_bus_sum = battery_store.loc[snapshot[0]].loc[snapshot[1]].groupby(battery_buses)
            hourly_battery_store_bus_sum =hourly_battery_store_bus_sum.sum("StorageUnit")

            LHS_Variable = hourly_renewable_power_bus_sum + hourly_battery_dispatch_bus_sum - hourly_battery_store_bus_sum
            
            #Now to add this as a constraint into the model 
            constraint_expressions_hourly_match = LHS_Variable >= hourly_match_candi_demand.loc[snapshot][region]
            print(constraint_expressions_hourly_match)
            m.add_constraints(constraint_expressions_hourly_match, name="Hourly_Match_{}_{}".format(snapshot,region))
            i += 1 
#hourly_matching_by_region()

#%%

initial_capacity = network.generators.p_nom.groupby(network.generators.carrier).sum()
initial_battery_capacity = network.storage_units.p_nom.sum()
initial_capacity = pd.concat([initial_capacity,pd.Series([initial_battery_capacity],index = ["Battery"])])
plt.bar(initial_capacity.index, initial_capacity)
plt.ylabel("Initial Capacity (MW)")

#%%

network.optimize.solve_model(method = 2, crossover =0, MIPGap = 0.1, IntFeasTol = 1e-4, FeasabilityTol = 1e-4, FeasRelax =1, solver_name = solver_name, MarkowitzTol=0.5)

print("All done! Well done champion\n")

print(network.statistics.curtailment(comps = None))

#%%

def capacity_results():

    capacity_output = pd.DataFrame()
    battery_output = pd.DataFrame()
    capacity = pd.DataFrame()
    capacity["2025"] = initial_capacity

    capacity_output["Region"] = network.generators.bus
    capacity_output["Technology"] = network.generators.carrier

    for period in network.investment_periods:
        capacity_output["Active_Status_{}".format(period)] = network.get_active_assets("Generator",period)
        capacity_output["p_nom_{}".format(period)] = network.generators.p_nom * capacity_output["Active_Status_{}".format(period)]
        capacity_output["p_nom_opt_{}".format(period)] = network.generators.p_nom_opt * capacity_output["Active_Status_{}".format(period)]

        battery_output["p_nom_opt_{}".format(period)] = network.storage_units.p_nom_opt.sum()

        plt.bar(capacity_output.groupby(capacity_output.Technology)["p_nom_opt_{}".format(period)].sum().index, capacity_output.groupby(capacity_output.Technology)["p_nom_opt_{}".format(period)].sum())
        plt.ylabel("p_nom_opt (MW)")
        plt.show()
            
        new_opt_capacity = capacity_output.groupby(capacity_output.Technology)["p_nom_opt_{}".format(period)].sum() - initial_capacity[:-1]
        plt.bar(new_opt_capacity.index, new_opt_capacity)
        plt.ylabel("New capacity (MW)")
        plt.show()

        capacity["{}".format(period)] = capacity_output.groupby(capacity_output.Technology)["p_nom_opt_{}".format(period)].sum()
        capacity["{}".format(period)]["Battery"] = network.storage_units.p_nom_opt.sum()
    #capacity = capacity.transpose()
    ax = capacity.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.xlabel("Year")
    plt.ylabel("Capacity (MW)")
    plt.title("Optimal Capacity by Technology")
    plt.show()

    #folder_path = 'Results_csvs'
    capacity.to_csv(os.path.join(folder_path, 'capacity.csv')) 
capacity_results()


#%%

def emissions_results():

    emissions_df = pd.DataFrame()

    emissions_df["Carrier"] = network.generators.carrier #Add in the carrier of each generator
    emissions_df["Region"] = network.generators.bus
    emissions_df["Emissions_Intensity"] = network.carriers.co2_emissions[network.generators.carrier].tolist() #Add in the emissions intensity of each carrier

    emissions_df["Generation"] = network.generators_t.p.sum() #Add in sum of all generation across optimisation period
    emissions_df["Emissions_Intensity"] = network.carriers.co2_emissions[network.generators.carrier].tolist() #Add in the emissions intensity of each carrier
    emissions_df["Emissions"] = emissions_df["Generation"]*emissions_df["Emissions_Intensity"] #Total emissions is generation * emissions intensity



    plt.bar(emissions_df.Emissions.groupby(emissions_df.Carrier).sum().index,emissions_df.Emissions.groupby(emissions_df.Carrier).sum())
    plt.ylabel("Total Emissions (tCO2eq")
    plt.show()

    emissions_intensity_by_region = emissions_df.Emissions.groupby(emissions_df.Region).sum()/emissions_df.Generation.groupby(emissions_df.Region).sum()

    grid_emissions_intensity = emissions_df.Emissions.sum()/emissions_df.Generation.sum()

    emissions_intensity_by_region = pd.concat([emissions_intensity_by_region,pd.Series([grid_emissions_intensity], index = ["NEM"])])
    plt.bar(emissions_intensity_by_region.index, emissions_intensity_by_region)
    plt.ylabel("Emissions Intensity (tCO2eq/MWh)")
    plt.show()
    
    #folder_path = 'Results_csvs'
    emissions_intensity_by_region.to_csv(os.path.join(folder_path, 'emissions.csv'))

    # for period in network.investment_periods:
    #     starthr = ((period-2030)/5)*8760
    #     endhr = ((period-2030)/5 + 1)*8760
    #     emissions_df["Generation_{}".format(period)] = network.generators_t.p[starthr:endhr].sum() #Add in sum of all generation across optimisation period
    #     emissions_df["Emissions_{}".format(period)] = emissions_df["Generation_{}".format(period)]*emissions_df["Emissions_Intensity"] #Total emissions is generation * emissions intensity
    #     emissions_intensity_by_region["{}".format(period)] = emissions_df.groupby(emissions_df.Region)["Emissions_{}".format(period)].sum()/emissions_df.groupby(emissions_df.Region)["Generation_{}".format(period)].sum()
emissions_results()

#%%
def generation_profile():

    for period in network.investment_periods:
        hrno = int((period - 2030)/5*8760) + 48
        colours = ["saddlebrown","black","brown", "orange", "blue", "yellow", "green"]
        ax1 = network.generators_t.p[hrno:(hrno+48)].groupby(network.generators.carrier, axis =1).sum().plot.area(color = colours)
        ax2 = ax1.twinx()
        ax2 = network.storage_units_t.state_of_charge[hrno:(hrno+48)].plot.line(ylabel = "Dispatch (MW)", ax = ax2)
        plt.legend().set_visible(False)
        plt.ylabel("Generation (MW)")
        plt.title(period)
        plt.show()
generation_profile()

#%%

def generation_profile_no_batteries():

    for period in network.investment_periods:
        hrno = int((period - 2030)/5*8760) + 48
        colours = ["saddlebrown","black","brown", "orange", "blue", "yellow", "green"]
        generation_mix = network.generators_t.p[hrno:(hrno+48)].groupby(network.generators.carrier, axis =1).sum()
        ax1 = generation_mix.plot.area(color = colours)
        #ax2 = ax1.twinx()
        #ax2 = network.storage_units_t.state_of_charge[hrno:(hrno+48)].plot.line(ylabel = "Dispatch (MW)", ax = ax2)
        plt.legend().set_visible(False)
        plt.ylabel("Generation (MW)")
        plt.title(period)
        plt.show()
        #folder_path = 'Results_csvs'
        generation_mix.to_csv(os.path.join(folder_path, 'generation_mix.csv'))

generation_profile_no_batteries()

#%%



# hourly_matching_by_region()
# def annual_regional_matching(): 
#             #matching_candi_consumption["non_candi_RPP_consumption"] = matching_candi_consumption["non_candi_RPP_consumpti    #100% Annual Matching Constraint, this doesn't distinguish by bus yet
#     for period in network.investment_periods:

#         #RHS - Needs to be a fixed value. Energy Consumed by CandI loads, plus the RPP 

#         period_start = int(((period - 2030)/5)*8760)
#         period_end  = int((((period - 2030)/5)+ 1)*8760)

#         matching_candi_consumption = pd.DataFrame()

#         matching_candi_consumption["bus"] = network.loads.bus[:10] #first 10 cols are the C&I matching loads
#         matching_candi_consumption["Matching_Consumption_(MWh)"] = network.loads_t.p_set.iloc[period_start:period_end, :10].sum() #2nd 10 loads are the net loads (that don't match)

#         non_matching_rpp_consumption = pd.DataFrame(network.loads_t.p_set.iloc[period_start:period_end, 10:].sum()) * rpp

#         #Need to reset indexes so we can add one column to the matching_candi_consumption dataframe
#         matching_candi_consumption = matching_candi_consumption.reset_index()
#         non_matching_rpp_consumption = non_matching_rpp_consumption.reset_index()

#         matching_candi_consumption['non_matching_rpp_consumption'] = non_matching_rpp_consumption.iloc[:, 1]
#         matching_candi_consumption = matching_candi_consumption.set_index('bus')

#         matching_candi_consumption["Net_RE_Consumption"] = matching_candi_consumption["Matching_Consumption_(MWh)"] + matching_candi_consumption["non_matching_rpp_consumption"]
        
#         for region in network.buses.index:
#         #LHS - Variables. Renewable Energy Generation, by region across the year

#             #First, return an x_array of the carrier/technology of each asset 
#             gen_carriers = network.generators.carrier*network.get_active_assets("Generator",period)
#             gen_carriers = gen_carriers.to_xarray()

#             is_hydro = gen_carriers == "Hydro"
#             is_solar = gen_carriers == "Solar"
#             is_wind = gen_carriers == "Wind"
#             is_renewable = is_hydro | is_solar | is_wind
#             gen_p = m.variables["Generator-p"]
#             renewable_power_variables = gen_p.where(is_renewable)
#             # double checking using pandas
#             # code below converts to pandas DataFrame, checks where the value is -1
#             # linopy fills locations where the condition is false with -1, so the list printed
#             # should only have renewable energy generators
#             test = renewable_power_variables.to_pandas()
#             test_2030 = test.loc[(period,)]
#             test_2030.where(test_2030 > 0).dropna(axis=1).columns.tolist()
#             # groupby buses
#             gen_buses = network.generators.bus.to_xarray()
#             #New COde
#             curr_gen_bus = gen_buses == region
#             bus_variables = gen_buses.where(curr_gen_bus)
#             # This line below gives you the sum of all renewable energy generators at each bus and for each hour
#             # So this should give you the ability to do regional hourly matching
#             hourly_renewable_power_bus_sum = renewable_power_variables.groupby(bus_variables).sum() #bus_variables used to be gen_buses
#             # Add `.sum("snapshot")` to produce annual LHS for each bus
#             annual_renewable_bus_power_sum = hourly_renewable_power_bus_sum.sum()

#             net_re_consumption_by_region = matching_candi_consumption.loc[region]["Net_RE_Consumption"]

#             constraint_expression_ann_match = annual_renewable_bus_power_sum >= net_re_consumption_by_region
#             m.add_constraints(constraint_expression_ann_match, name="100% RES_{}_{}".format(region,period))

#annual_regional_matching()

#%%

# def hourly_matching_by_NEM():
#     #RHS - Need to get a DF where each row represents a bus, and each 
#     hourly_match_candi_demand = pd.DataFrame()
#     hourly_match_candi_demand = network.loads_t.p_set.iloc[:,:10] * CFE_score
#     #Rename column names to be the same as the buses
#     hourly_match_candi_demand.columns = [col.split()[0] for col in hourly_match_candi_demand.columns]

#     #LHS - Variables are Generation from RE assets, + dispatch - storage of batteries
#     for snapshot in network.snapshots[:2]:

#         for region in network.buses.index:
#             #FIRST - RE Generation variable set up 
#             gen_carriers = network.generators.carrier*network.get_active_assets("Generator",snapshot[0])
#             gen_carriers = gen_carriers.to_xarray()

#             is_hydro = gen_carriers == "Hydro"
#             is_solar = gen_carriers == "Solar"
#             is_wind = gen_carriers == "Wind"

#             is_renewable = is_hydro | is_solar | is_wind
#             gen_p = m.variables["Generator-p"]
#             renewable_power_variables = gen_p.where(is_renewable)
#             gen_buses = network.generators.bus.to_xarray()
#             curr_gen_bus = gen_buses == region
#             bus_variables = gen_buses.where(curr_gen_bus)

#             hourly_renewable_power_bus_sum = renewable_power_variables.groupby(bus_variables).sum()

#             #SECOND - Battery dispatch variable set up 
#             battery_dispatch = m.variables["StorageUnit-p_dispatch"]
#             battery_store = m.variables["StorageUnit-p_store"]

#             battery_buses = network.storage_units.bus.to_xarray()
#             curr_batt_bus = battery_buses == region
#             battery_buses = battery_buses.where(curr_batt_bus)
            
#             hourly_battery_dispatch_bus_sum = battery_dispatch.groupby(battery_buses).sum()
#             hourly_battery_store_bus_sum = battery_store.groupby(battery_buses).sum()

#             LHS_Variable = hourly_renewable_power_bus_sum + hourly_battery_dispatch_bus_sum - hourly_battery_store_bus_sum
            
#             #Now to add this as a constraint into the model 
#             constraint_expressions_hourly_match = LHS_Variable >= hourly_match_candi_demand.loc[snapshot][region]
#             m.add_constraints(constraint_expressions_hourly_match, name="Hourly_Match_{}_{}".format(snapshot,region))

#    # for snapshot in network.snapshots.get_level_values("period") == period
#             #Renewable Energy Generation, by region, for every hour
#             #First, return an x_array of the carrier/technology of each asset 
           
# hourly_matching_by_NEM()

# # %%
