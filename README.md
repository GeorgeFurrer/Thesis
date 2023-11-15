# Thesis
Modelling System Wide Effects of 24/7 Corporate PPAs on the NEM

This github repo is for a PyPSA model fo the NEM. The model can perform a capacity expansion of the NEM for the years 2030, 2035 and 2040. The user can add constraints to test different research questions. 
This model is currently set up to analyse the capacity, emissions and curtailment pending whether C&Is match their consumption with renewable energy on an annual or hourly basis. 

Some of the packages the user will need to have downloaded: 
 - PyPSA package
 - Gurobi or equivalent solver, with license

When pulling data, the user will need to create two folders in the working directory named "00_Capacity_csvs" and "01_Emissions_csvs"
User can change the "Portion of matching C&I's" (any value between 0 and 1) 
Simply uncomment out the functions that the user wishes to use (annual or hourly matching, grouping over the whole NEM or by region)
