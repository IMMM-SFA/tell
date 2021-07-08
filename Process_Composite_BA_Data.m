% Process_Composite_BA_Data.m
% 20210415
% Casey D. Burleyson
% Pacific Northwest National Laboratory

% This script takes the time-series of average meteorology, total population, and load 
% in a given balancing authority (BA) and combines all of the data together into a single array. 
% The needed input files are stored on PIC at: /projects/im3/tell/.The output file format is 
% given below. This script coresponds to needed functionality 1.5 on this Confluence page:
% https://immm-sfa.atlassian.net/wiki/spaces/IP/pages/1732050973/2021-02-22+TELL+Meeting+Notes.
% The script relies on one function that provides BA metadata based on the
% EIA BA number: EIA_930_BA_Information_From_BA_Number.m. All times are in UTC. 
% Missing values are reported as -9999 in the .csv output files and as NaNs in the .mat output files.
%
%   .mat output file format:
%   C1:  MATLAB date number
%   C2:  Year
%   C3:  Month
%   C4:  Day
%   C5:  Hour
%   C6:  U.S. Census Bureau total population in the counties the BA covers
%   C7:  Population-weighted NLDAS temperature in K
%   C8:  Population-weighted NLDAS specific humidity in kg/kg
%   C9:  Population-weighted NLDAS downwelling shortwave radiative flux in W/m^2
%   C10: Population-weighted NLDAS downwelling longwave radiative flux in W/m^2
%   C11: Population-weighted NLDAS wind speed in m/s
%   C12: EIA 930 forecast demand in MWh
%   C13: EIA 930 adjusted demand in MWh
%   C14: EIA 930 adjusted generation in MWh
%   C15: EIA 930 adjusted net interchange with adjacent balancing authorities in MWh
%
%   .csv output file format: 
%   C1:  Year
%   C2:  Month
%   C3:  Day
%   C4:  Hour
%   C5:  U.S. Census Bureau total population in the counties the BA covers
%   C6:  Population-weighted NLDAS temperature in K
%   C7:  Population-weighted NLDAS specific humidity in kg/kg
%   C8:  Population-weighted NLDAS downwelling shortwave radiative flux in W/m^2
%   C9:  Population-weighted NLDAS downwelling longwave radiative flux in W/m^2
%   C10: Population-weighted NLDAS wind speed in m/s
%   C11: EIA 930 adjusted forecast demand in MWh
%   C12: EIA 930 adjusted demand in MWh
%   C13: EIA 930 adjusted generation in MWh
%   C14: EIA 930 adjusted net interchange with adjacent balancing authorities in MWh

warning off all; clear all; close all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              BEGIN USER INPUT SECTION               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the data input and output directories:
load_data_input_dir = 'C:/Users/mcgr323/projects/tell/raw_data/inputs/BA_Hourly_Load/Matlab_Files/';
population_data_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/BA_Hourly_Population/Matlab_Files/';
meteorology_data_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/BA_Hourly_Meteorology/Matlab_Files/';
mat_data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/Composite_BA_Hourly_Data/Matlab_Files/';
csv_data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/Composite_BA_Hourly_Data/CSV_Files/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              END USER INPUT SECTION                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              BEGIN PROCESSING SECTION               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make a list of all of the files in the meteorology directory and index the list
% with the year and EIA BA # so that it can be mapped to the other variables:
input_files = dir([meteorology_data_input_dir,'*.mat']);
for file = 1:size(input_files,1)
    filename = input_files(file,1).name;
    year = str2num(filename(1,(size(filename,2)-7):(size(filename,2)-4)));
    BA_Short_Name = filename(1,1:(size(filename,2)-33));
    [EIA_BA_Number,BA_Long_Name] = EIA_930_BA_Information_From_BA_Short_Name(BA_Short_Name);
    Met_Names(file,1).filename = filename;
    Met_Times(file,1) = year;
    Met_Times(file,2) = EIA_BA_Number;
    clear filename year BA_Short_Name EIA_BA_Number BA_Long_Name
end
clear input_files file

% Make a list of all of the files in the population directory and index the list
% with the year and EIA BA # so that it can be mapped to the other variables:
input_files = dir([population_data_input_dir,'*.mat']);
for file = 1:size(input_files,1)
    filename = input_files(file,1).name;
    BA_Short_Name = filename(1,1:(size(filename,2)-27));
    [EIA_BA_Number,BA_Long_Name] = EIA_930_BA_Information_From_BA_Short_Name(BA_Short_Name);
    Pop_Names(file,1).filename = filename;
    Pop_Times(file,1) = EIA_BA_Number;
    clear filename year BA_Short_Name EIA_BA_Number BA_Long_Name
 end
 clear input_files file

% Make a list of all of the files in the load directory and index the list
% with the year and EIA BA # so that it can be mapped to the other variables:
input_files = dir([load_data_input_dir,'*.mat']);
for file = 1:size(input_files,1)
    filename = input_files(file,1).name;
    BA_Short_Name = filename(1,1:(size(filename,2)-21));
    [EIA_BA_Number,BA_Long_Name] = EIA_930_BA_Information_From_BA_Short_Name(BA_Short_Name);
    Load_Names(file,1).filename = filename;
    Load_Times(file,1) = EIA_BA_Number;
    clear filename year BA_Short_Name EIA_BA_Number BA_Long_Name
end
clear input_files file
   
% Create a time vector that has an hourly interval between the start of 2015 and the end of 2020:
target_time_vector = [datenum(2015,1,1,0,0,0):(1/24):datenum(2020,12,31,23,0,0)]';

% Create a list of all of the unique BAs based on the data in the meteorology directory:
unique_bas = unique(Met_Times(:,2));

% Loop through the list of unique BAs and merge the meteorology, population, and load data into a common time-series.
for i = 1:size(unique_bas,1)
    % Display the progress:
    Progress = 100.*(i/size(unique_bas,1))
    
    % Retrieve the BA code to be used in the output file:
    [BA_Code,BA_Long_Name] = EIA_930_BA_Information_From_BA_Number(unique_bas(i,1));
    
    % If there is a corresponding load file and population file then try to merge the data:
    if isempty(find(Load_Times(:,1) == unique_bas(i,1))) == 0 & isempty(find(Pop_Times(:,1) == unique_bas(i,1))) == 0
       
       % Subset the meteorology files to only that BA:
       Met_Names_Subset = Met_Names(find(Met_Times(:,2) == unique_bas(i,1)),:);
       Met_Times_Subset = Met_Times(find(Met_Times(:,2) == unique_bas(i,1)),:);
       % Loop over the meteorology files for that BA and concatenate the data into a single array:
       for file = 1:size(Met_Times_Subset,1)
           load([meteorology_data_input_dir,Met_Names_Subset(file,1).filename]);
           if file == 1
              Meteorology = Data;
           else
              Meteorology = cat(1,Meteorology,Data);
           end
           clear Data
       end
       % Clugey fix to work around some bug in the datenum function in Matlab:
       Meteorology(:,1) = datenum(Meteorology(:,2),Meteorology(:,3),Meteorology(:,4),Meteorology(:,5),0,0);
       clear file Met_Names_Subset Met_Times_Subset
       
       % Load in the population data for the BA:
       load([population_data_input_dir,Pop_Names(find(Pop_Times(:,1) == unique_bas(i,1)),1).filename]);
       Population = Data;
       % Clugey fix to work around some bug in the datenum function in Matlab:
       Population(:,1) = datenum(Population(:,2),Population(:,3),Population(:,4),Population(:,5),0,0);
       clear Data
       
       % Load in the load data for the BA:
       load([load_data_input_dir,Load_Names(find(Load_Times(:,1) == unique_bas(i,1)),1).filename]);
       Load = Data;
       % Clugey fix to work around some bug in the datenum function in Matlab:
       Load(:,1) = datenum(Load(:,2),Load(:,3),Load(:,4),Load(:,5),0,0);
       clear Data
          
       % Create a new array starting from the target time vector:
       Data(:,1) = target_time_vector(:,1);
       Data(:,2:7) = datevec(target_time_vector(:,1));
       % Clugey fix to work around some bug in the datenum function in Matlab:
       Data(:,1) = datenum(Data(:,2),Data(:,3),Data(:,4),Data(:,5),0,0);
       Data = Data(:,1:5);
   
       % Loop over the new array and if there is a matching date/time
       % within the population, meteorology, and load data then pull that
       % matching data into the new array:
       for row = 1:size(Data,1)
           if isempty(find(Population(:,1) == Data(row,1))) == 0
              Data(row,6) = Population(find(Population(:,1) == Data(row,1)),6);
           else
              Data(row,6) = NaN.*0;
           end
           if isempty(find(Meteorology(:,1) == Data(row,1))) == 0
              Data(row,7:11) = Meteorology(find(Meteorology(:,1) == Data(row,1)),6:10);
           else
              Data(row,7:11) = NaN.*0;
           end
           if isempty(find(Load(:,1) == Data(row,1))) == 0
              Data(row,12:15) = Load(find(Load(:,1) == Data(row,1)),6:9);
           else
              Data(row,12:15) = NaN.*0;
           end
       end
       clear row

       % Save the data as a .mat file:
       save([mat_data_output_dir,BA_Code,'_Hourly_Composite_Data_2015_to_2020.mat'],'Data');

       % Convert the data into a table and save it as a .csv file:
       Data(find(isnan(Data) == 1)) = -9999;
       Output_Table = array2table(Data(:,2:15));
       Output_Table.Properties.VariableNames = {'Year','Month','Day','Hour','Population','Temperature','Specific_Humidity','Shortwave_Radiation','Longwave_Radiation','Wind_Speed','Forecast_Demand','Demand','Generation','Interchange'};
       writetable(Output_Table,strcat([csv_data_output_dir,BA_Code,'_Hourly_Composite_Data_2015_to_2020.csv']),'Delimiter',',','WriteVariableNames',1);
       clear Data Output_Table
    end
    clear BA_Code BA_Long_Name Meteorology Load Population Progress
end
clear unique_bas i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               END PROCESSING SECTION                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                BEGIN CLEANUP SECTION                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear load_data_input_dir meteorology_data_input_dir population_data_input_dir csv_data_output_dir mat_data_output_dir target_time_vector Load_Names Load_Times Met_Names Met_Times Pop_Names Pop_Times
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 END CLEANUP SECTION                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%