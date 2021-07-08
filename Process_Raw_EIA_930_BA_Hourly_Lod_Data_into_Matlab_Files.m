% Process_Raw_EIA_930_BA_Hourly_Load_Data_into_Matlab_Files.m
% 20210413
% Casey D. Burleyson
% Pacific Northwest National Laboratory

% Take the raw EIA-930 hourly load data by balancing authority
% and convert it from .xlsx files into .mat and .csv files. The output
% file format is given below. All times are in UTC. Missing values are reported as -9999
% in the .csv output files and are reported as NaN in the .mat output files. 
% This script coresponds to needed functionality 1.2 on this Confluence page:
% https://immm-sfa.atlassian.net/wiki/spaces/IP/pages/1732050973/2021-02-22+TELL+Meeting+Notes.
% The raw data used as input to this script is stored on PIC at
% /projects/im3/tell/raw_data/EIA_930/.
%
%   .mat output file format: 
%   C1: MATLAB date number
%   C2: Year
%   C3: Month
%   C4: Day
%   C5: Hour
%   C6: Forecast demand in MWh
%   C7: Adjusted demand in MWh
%   C8: Adjusted generation in MWh
%   C9: Adjusted net interchange with adjacent balancing authorities in MWh
%
%   .csv output file format: 
%   C1: Year
%   C2: Month
%   C3: Day
%   C4: Hour
%   C5: Forecast demand in MWh
%   C6: Adjusted demand in MWh
%   C7: Adjusted generation in MWh
%   C8: Adjusted net interchange with adjacent balancing authorities in MWh

warning off all; clear all; close all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              BEGIN USER INPUT SECTION               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the data input and output directories:
data_input_dir = 'C:/Users/mcgr323/projects/tell/raw_data/raw_data/EIA_930/Balancing_Authority/';
mat_data_output_dir = 'C:/Users/mcgr323/projects/tell/raw_data/inputs/BA_Hourly_Load/Matlab_Files/';
csv_data_output_dir = 'C:/Users/mcgr323/projects/tell/raw_data/inputs/BA_Hourly_Load/CSV_Files/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              END USER INPUT SECTION                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              BEGIN PROCESSING SECTION               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make a list of all of the files in the data_input_dir:
input_files = dir([data_input_dir,'*.xlsx']);

% Loop over each of the files and extract the variables of interest:
for file = 1:size(input_files,1)
    % Read in the raw .xlsx file:
    filename = input_files(file,1).name;
    [~,~,Raw_Data] = xlsread([data_input_dir,filename],'Published Hourly Data');
    Raw_Data(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),Raw_Data)) = {''};
    
    % Extract the balancing authority code:
    BA_Code = char(Raw_Data{2,1});
    
    % Loop over the rows and extract the variables of interest:
    counter = 0;
    for row = 2:size(Raw_Data,1)
        counter = counter + 1;
        
        % Convert the UTC date from the Excel time format to the Matlab date number format:
        Data(counter,1) = datenum(Raw_Data{row,2} + 693960);
        Data(counter,2:7) = datevec(Raw_Data{row,2} + 693960);
        
        % Extract the forecast demand:
        if isempty(Raw_Data{row,8}) == 0
           Data(counter,6) = Raw_Data{row,8};
        else
           Data(counter,6) = NaN.*0;
        end
        
        % Extract the adjusted demand:
        if isempty(Raw_Data{row,15}) == 0
           Data(counter,7) = Raw_Data{row,15};
        else
           Data(counter,7) = NaN.*0;
        end
        
        % Extract the adjusted generation:
        if isempty(Raw_Data{row,16}) == 0
           Data(counter,8) = Raw_Data{row,16};
        else
           Data(counter,8) = NaN.*0;
        end
        
        % Extract the adjusted interchange from adjacent balancing authorities:
        if isempty(Raw_Data{row,17}) == 0
           Data(counter,9) = Raw_Data{row,17};
        else
           Data(counter,9) = NaN.*0;
        end
    end
        
    % Save the data as a .mat file:
    save([mat_data_output_dir,BA_Code,'_Hourly_Load_Data.mat'],'Data');
    
    % Convert the NaN values to -9999 for the .csv file:
    Data(find(isnan(Data) == 1)) = -9999;
    
    % Convert the data into a table and save it as a .csv file:
    Output_Table = array2table(Data(:,2:9));
    Output_Table.Properties.VariableNames = {'Year','Month','Day','Hour','Forecast_Demand_MWh','Adjusted_Demand_MWh','Adjusted_Generation_MWh','Adjusted_Interchange_MWh'};
    writetable(Output_Table,strcat([csv_data_output_dir,BA_Code,'_Hourly_Load_Data.csv']),'Delimiter',',','WriteVariableNames',1);
    clear Output_Table Data filename Raw_Data counter row BA_Code
end
clear input_files file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               END PROCESSING SECTION                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                BEGIN CLEANUP SECTION                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear data_input_dir mat_data_output_dir csv_data_output_dir
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 END CLEANUP SECTION                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%