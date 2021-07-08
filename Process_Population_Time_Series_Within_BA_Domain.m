% Process_Population_Time_Series_Within_BA_Domain.m
% 20210413
% Casey D. Burleyson
% Pacific Northwest National Laboratory

% This script takes as .mat files containing the county mapping of
% utilities and balancing authorities (BAs) and computes the annual
% total population in counties covered by that BA. Those populations
% are then interpolated to an hourly resolution in order to match the 
% temporal resolution of the load and meteorology time series.
% The output file format is given below. The script takes as input the 
% years to process as well as paths to the relevant input and output directories. 
% All of the required input files are stored on PIC at /projects/im3/tell/inputs/.
% The script relies on one function that provides BA metadata based on the
% EIA BA number: EIA_930_BA_Information_From_BA_Number.m. 
% This script coresponds to needed functionality 1.4 on this Confluence page:
% https://immm-sfa.atlassian.net/wiki/spaces/IP/pages/1732050973/2021-02-22+TELL+Meeting+Notes.
%
%   .mat output file format: 
%   C1: MATLAB date number
%   C2: Year
%   C3: Month
%   C4: Day
%   C5: Hour
%   C6: Total population within the BA domain
%
%   .csv output file format: 
%   C1: Year
%   C2: Month
%   C3: Day
%   C4: Hour
%   C5: Total population within the BA domain

warning off all; clear all; close all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              BEGIN USER INPUT SECTION               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set some processing flags:
start_year = 2015; % Starting year of time series
end_year = 2019; % Ending year of time series

% Set the data input and output directories:
population_data_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/';
service_territory_data_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/Utility_Mapping/Matlab_Files/';
mat_data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/BA_Hourly_Population/Matlab_Files/';
csv_data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/BA_Hourly_Population/CSV_Files/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              END USER INPUT SECTION                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              BEGIN PROCESSING SECTION               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load in the county population dataset used to population-weight the meteorology variables:
load([population_data_input_dir,'county_populations_2000_to_2019.mat']);

% Loop over all of the years in the date range provided in the user input section:
counter = 0;
for year = start_year:1:end_year
    % Load in the BA service territory dataset:
    load([service_territory_data_input_dir,'BA_Service_Territory_',num2str(year),'.mat']);

    % Loop over all of the BAs with a valid service territory sum the population for all counties in the BA territory:
    for i = 1:size(BA_Metadata,1)
        Territory = BA_Service_Territory(find(BA_Service_Territory(:,2) == BA_Metadata{i,1}),:);
       
        if isempty(Territory) == 0
           counter = counter + 1;
           
           % Assign the population to each county in the BA territory:
           Territory = unique(Territory(:,1),'rows');
           for row = 1:size(Territory,1)
               Territory(row,2) = Population(find(FIPS_Code(:,1) == Territory(row,1)),find(Year(1,:) == year));
           end
           clear row
           
           % Sum the populations and put the output in a common data array:
           All_Data(counter,1) = year;
           All_Data(counter,2) = BA_Metadata{i,1};
           All_Data(counter,3) = nansum(Territory(:,2));
        end
        clear Progress Territory
    end
    clear ans i BA_Metadata BA_Service_Territory
end
clear year counter Year Population FIPS_Code

% Create the hourly time vector to interpolate to:
target_time_vector = [datenum(start_year,1,1,0,0,0):(1/24):datenum(end_year,12,31,23,0,0)];

% Loop over all the unique BAs and interpolate the yearly population to an hourly value based on the target_time_vector defined above:
unique_bas = unique(All_Data(:,2));
for i = 1:size(unique_bas,1)
    % Look up the BA code based on the unique EIA BA number:
    [BA_Code,BA_Long_Name] = EIA_930_BA_Information_From_BA_Number(unique_bas(i,1));
    
    % Subset the data and do the interpolation:
    Subset = All_Data(find(All_Data(:,2) == unique_bas(i,1)),:);
    Subset(:,4) = datenum(Subset(:,1),1,1,0,0,0);
    Data(:,1) = target_time_vector';
    Data(:,2:7) = datevec(Data(:,1));
    Data(:,6) = round(interp1(Subset(:,4),Subset(:,3),target_time_vector,'linear','extrap'))';
    Data = Data(:,1:6);
    
    % Save the data as a .mat file:
    save([mat_data_output_dir,BA_Code,'_Hourly_Population_Data.mat'],'Data');
    
    % Convert the data into a table and save it as a .csv file:
    Output_Table = array2table(Data(:,2:6));
    Output_Table.Properties.VariableNames = {'Year','Month','Day','Hour','Population'};
    writetable(Output_Table,strcat([csv_data_output_dir,BA_Code,'_Hourly_Population_Data.csv']),'Delimiter',',','WriteVariableNames',1);
    clear Output_Table Data BA_Code BA_Long_Name Subset
end
clear i unique_bas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               END PROCESSING SECTION                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                BEGIN CLEANUP SECTION                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear data_output_dir population_dir service_territory_dir target_time_vector 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 END CLEANUP SECTION                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%