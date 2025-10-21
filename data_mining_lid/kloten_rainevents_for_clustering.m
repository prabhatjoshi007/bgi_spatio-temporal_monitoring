%%
% Description:
%   This MATLAB script processes and analyzes rainfall and underdrain flow
%   data from the Kloten dataset. It performs tasks including:
%     - Reading input timetables from CSV files for both "Unclogged" and
%       "Clogged" BGI conditions.
%     - Generating time series figures that compare underdrain flow and rainfall.
%     - Filtering data using a moving average (movmean).
%     - Segmenting rainfall data into monthly subsets.
%     - Clustering monthly rainfall characteristics via k‐means (using a gap
%       statistic for optimal cluster determination).
%     - Detecting and summarizing individual rainfall events (computing metrics
%       such as mean, maximum, total precipitation, duration, etc.).
%     - Saving figures and output tables to designated directories.
%
% User Inputs:
%   - Path to the main directory
%   - createTSfigure: Flag to decide whether to generate time series (TS) figures.
%   - writeTT: Flag to decide whether to write the timetables (rainfall, runoff,
%              and soil moisture) to file.
%   - bgi_condition: Choice between BGI conditions – 1 for "Unclogged" and 2 for
%                    "Clogged".
%   - rain_segment: Flag to enable segmentation of rainfall data.
%
%
% Dependencies:
%   MATLAB Built-in Functions:
%     - Workspace management: clear, clc, close all.
%     - File and directory handling: cd, readtimetable, writetimetable, writetable.
%     - Plotting and visualization: figure, plot, fill, text, xlabel, ylabel, xline,
%       grid, savefig, exportgraphics.
%     - Data processing: movmean, std, max, min, range, unique, year, month, retime,
%       normalize.
%     - Clustering and evaluation: evalclusters, kmeans.
%   Toolboxes:
%     - Statistics and Machine Learning Toolbox (for clustering functions such as
%       kmeans and evalclusters).
%   External Files:
%     - Input CSV Files:
%         • kloten_br_basicmodel_cr_0.csv (for the "Unclogged" condition)
%         • kloten_br_basicmodel_cr_750.csv (for the "Clogged" condition)%
%
% Outputs:
%   - Time series figures saved as MATLAB figure (.fig) and JPEG (.jpg) files.
%   - Written timetables for rainfall, runoff, and soil moisture (if enabled).
%   - A figure with segmented rainfall events and cluster annotations.
%   - An event summary text file (Kloten_EventSummary_B{boxNumber}_{startYear}_{endYear}.txt)
%     containing computed statistics for each detected rainfall event.
%
% Usage:
%   1. Place the required CSV files in the designated input directory.
%   2. Adjust the hard-coded directory paths if necessary.
%   3. Run the script in MATLAB.
%   4. Follow the prompts to select analysis options (e.g., generating figures,
%      writing outputs, selecting BGI condition, segmentation, period selection).
%   5. Review the generated figures and output tables for further analysis.
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
clear
clc
close all

%%
% Prompt the user to select the main folder (Data_repository)
mainFolder = uigetdir(pwd, 'Select the Main Folder');
cd(mainFolder)

if mainFolder == 0
    disp('No folder selected. Operation canceled.');
else
    % Generate a path that includes all subfolders
    allFolders = genpath(mainFolder);

    % Add the paths to MATLAB
    addpath(allFolders);

    % Confirm
    disp(['All files and subfolders from "', mainFolder, '" have been added to the path.']);
end


%% User inputs

createTSfigure = input('Do you want to create TS figures? 1: Y, other: N:: ');
writeTT = input('Do you want to write the timetables? 1-->Y, 2-->N: ');
bgi_condition = input('What is the BGI condition? 1--> unclogged; 2--> clogged: ');
movmean_window = 1;
rain_segment = input('Do you want to segment rainfall? 1--> Y, else-->N: ');
boxNumber = 1;

%%
Kloten_BGI_unclogged = readtimetable('no_clog.csv');

Kloten_BGI_clogged = readtimetable('clog.csv');

Kloten_BGI_rain = readtimetable('Kloten_B1_rainfall.txt');

Kloten_BGI_unclogged.Properties.DimensionNames{1} = 'Time';
Kloten_BGI_clogged.Properties.DimensionNames{1} = 'Time';

st_time = datetime('1986-01-03 00:00');
end_time = datetime('1996-12-30 23:50');
tr = timerange(st_time, end_time, 'closed');

Kloten_BGI_unclogged = Kloten_BGI_unclogged(tr, :);
Kloten_BGI_clogged = Kloten_BGI_clogged(tr, :);
Kloten_rain = Kloten_BGI_rain(tr, :); 


%%

earlyYears_start = datetime(1986, 01, 03, 0, 0, 0); % Start datetime for shaded area
earlyYears_end = datetime(1990, 12, 28, 23, 50, 0); % End datetime for shaded area
earlyYears_x_shade = [earlyYears_start earlyYears_end earlyYears_end earlyYears_start];
earlyYears_y_shade = [0 0 200 200] ;

laterYears_start = datetime(1991, 01, 03, 0, 0, 0); % Start datetime for shaded area
laterYears_end = datetime(1996, 12, 30, 23, 50, 0); % End datetime for shaded area
laterYears_x_shade = [laterYears_start laterYears_end laterYears_end laterYears_start];
laterYears_y_shade = [0 0 200 200];


if createTSfigure == 1

    figure(18)
    yyaxis left;
    fill(earlyYears_x_shade, earlyYears_y_shade, [0.4660 0.6740 0.1880], 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off')
    hold on
    plot(Kloten_BGI_clogged.Time, Kloten_BGI_clogged.Drain_outflow_mmh_1, '.-', 'LineWidth', 2, 'Color','#2c7fb8', ...
        DisplayName = 'Clogged');
    hold on
    plot(Kloten_BGI_clogged.Time, Kloten_BGI_unclogged.Drain_outflow_mmh_1, '-', 'LineWidth', 2, 'Color','#e6550d', ...
        DisplayName = 'Unclogged');
    ylim([0 50])
    text(datetime(1987, 01, 01, 0, 0, 0),30,{'Early years'},'FontSize', 38, 'Color',	'k', 'FontName','Calibri')
    ylabel({'Underdrain Flow (UF)', '[mm.h-1]'});
    xlabel('Year');
    %ylim([0 5])
    set(gca, 'YColor', 'k'); % Set the color of the left y-axis to blue
    set(gca, 'FontSize',20)
    grid on;


    % Plot the rainfall on the right y-axis with reversed axis
    yyaxis right;
    fill(laterYears_x_shade, laterYears_y_shade, [0.9290 0.905 0.949], 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off')
    fill(laterYears_x_shade, laterYears_y_shade, [0.9290 0.6940 0.1250], 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off')
    hold on
    plot(Kloten_rain.Time, Kloten_rain.Rain, '-', 'LineWidth', 1.5, 'Color','#999999', ...
        DisplayName = 'Rainfall');
    hold on
    ylabel({'Rainfall', '[mm.h-1]'});
    xlabel('Year')
    text(datetime(1994, 01, 01, 0, 0, 0),100,{'Later years'},'FontSize', 38, 'Color',	'k', 'FontName','Calibri')
    %text(datetime(2008, 01, 01, 0, 0, 0),100,{'Later years II'},'FontSize', 38, 'Color',	'k', 'FontName','Calibri')
    ylim([0 200])
    set(gca, 'YColor', 'k'); % Set the color of the left y-axis to blue
    set(gca, 'YDir', 'reverse'); % Reverse the y-axis
    set(gca, 'FontSize',20)
    legend('FontSize',20, 'Location','northeast')
    fontname(gcf,"Calibri")
    set(figure(18), 'WindowState', 'maximized');
    fontname(gcf,"Calibri")

    savefig(gcf,'_figures\_kloten\rain_uf_ts_clogged_and_unclogged.fig')
    exportgraphics(gcf, '_figures\_kloten\rain_uf_ts_clogged_and_unclogged.jpg', 'Resolution',800)
end

%%

if bgi_condition == 1
    Kloten_bgi = Kloten_BGI_unclogged;
    condition = 'unclogged';

else
    Kloten_bgi = Kloten_BGI_clogged;
    condition = 'clogged';
end

Kloten_inflow = Kloten_bgi(:,2);
Kloten_runoff = Kloten_bgi(:,9);
Kloten_SM = Kloten_bgi(:,12);

Kloten_rain.Properties.DimensionNames{1} = 'time';
Kloten_rain.Properties.VariableNames{1} = 'rain_mm_h-1';

Kloten_inflow.Properties.DimensionNames{1} = 'time';
Kloten_inflow.Properties.VariableNames{1} = 'inflow_mm_h-1';

Kloten_runoff.Properties.DimensionNames{1} = 'time';
Kloten_runoff.Properties.VariableNames{1} = 'runoff_mm_h-1';

Kloten_SM.Properties.DimensionNames{1} = 'time';
Kloten_SM.Properties.VariableNames{1} = 'soil_moisture';


%%

Kloten_runoff.("runoff_mm_h-1") = movmean(Kloten_runoff.("runoff_mm_h-1"), movmean_window);
Kloten_inflow.("inflow_mm_h-1") = movmean(Kloten_inflow.("inflow_mm_h-1"), movmean_window);
Kloten_rain.("rain_mm_h-1") = movmean(Kloten_rain.("rain_mm_h-1"), movmean_window);


if writeTT == 1
    cd(fullfile(mainFolder, '_processed_data/_kloten'))
    writetimetable(Kloten_inflow, sprintf('kloten_%s_b%d_inflow.csv',condition, boxNumber));
    writetimetable(Kloten_rain, sprintf('kloten_%s_b%d_rainfall.csv',condition, boxNumber));
    writetimetable(Kloten_runoff,sprintf('kloten_%s_b%d_runoff.csv',condition, boxNumber));
    writetimetable(Kloten_SM,sprintf('kloten_%s_b%d_soilmoisture.csv',condition, boxNumber));
end


%%
if rain_segment == 1
    % Initialize a cell array to hold the monthly data for each year
    years = year(Kloten_rain.time);
    uniqueYears = unique(years);
    monthlyData = cell(length(uniqueYears), 12);

    % Loop through each year and month to extract data
    for i = 1:length(uniqueYears)
        currentYear = uniqueYears(i);
        for mm = 1:12
            % Extract data for the current year and month
            currentMonthData = Kloten_rain(year(Kloten_rain.time) == currentYear & month(Kloten_rain.time) == mm, :);


            % Store the current month's data in the cell array
            monthlyData{i, mm} = currentMonthData;
            monthly_std(i,mm) = std(monthlyData{i,mm}.("rain_mm_h-1"));

            if isempty(monthlyData{i,mm}.("rain_mm_h-1"))
                monthly_max(i,mm) = nan;
            else
                monthly_max(i,mm) = max(monthlyData{i,mm}.("rain_mm_h-1"));
            end
        end
    end

    rain_mean = retime(Kloten_rain,'monthly','mean');
    rain_max = retime(Kloten_rain,'monthly','max');
    rain_min = retime(Kloten_rain, 'monthly', 'min');
    rain_range = rain_max.("rain_mm_h-1") - rain_min.("rain_mm_h-1");
    rain_std = monthly_std(:);
    rain_std = rain_std(~isnan(rain_std));

    rain_array = [rain_mean.("rain_mm_h-1") rain_max.("rain_mm_h-1") rain_range rain_std];

    scaled_data = normalize(rain_array);


    %%
    eval_result = evalclusters(scaled_data, 'kmeans', 'gap', 'KList', 2:10);
    optimal_k = eval_result.OptimalK;

    [idx, C] = kmeans(scaled_data, optimal_k, 'Replicates', 10, 'Display', 'final');


    %%
    % Add cluster index to the table
    rain_mean.Cluster = idx;
    rain_mean.Month = month(rain_mean.time);


    %%
    segments = zeros(12,optimal_k);

    for m = 1:12
        aaa{:,m} = rain_mean.Cluster(rain_mean.Month == m);
    end

    for m = 1:12
        for c = 1:optimal_k
            segments(m,c) = sum(and(rain_mean.Month == m,  rain_mean.Cluster == c));
        end
    end

    segments_sum = sum(segments,2);

    for i = 1:length(segments_sum)
        for j = 1:size(segments,2)
            segments_pc(i,j) = segments(i,j)/segments_sum(i) * 100;
        end
    end
    %%
    mult = 1; % to convert from mm.(5 min)-1 to mm.h-1
    for j = 1:12
        mean_of_monthly_max(1,j) = mean(monthly_max(~isnan(monthly_max(:,j)),j));
    end

    monthly_max(:,13) = monthly_max(:,1);
    mean_of_monthly_max(1,13) = mean_of_monthly_max(1,1);

    for i = 1:size(monthly_max, 1)

        figure(5)

        plot(monthly_max(i,:) * mult,'-x','Color','#fdcc8a','LineWidth',0.5,'HandleVisibility','off')
        hold on

    end
    plot(mean_of_monthly_max * mult,'k-o','LineWidth',2,'DisplayName','Mean of max.')
    ylabel('Max. monthly rainfall [mm.h-1]')
    xlabel('Month')
    xticks(1:13)
    xlim([1 13])
    xticklabels({'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '1'})
    legend()
    set(gca,'FontSize', 35)
    grid on;
    hold off
    set(figure(5), 'WindowState', 'maximized');

end



%%
temp_res = 10; %[min]
interevent_dur = 6; %[h]
min_sum = 2; %[mm]

Kloten_rain.Event_1 = zeros(size(Kloten_rain,1),1);
Kloten_rain.Event_2 = zeros(size(Kloten_rain,1),1);
Kloten_rain.Event_3 = zeros(size(Kloten_rain,1),1);

indx = movsum(Kloten_rain.("rain_mm_h-1") * temp_res/60, (interevent_dur * 60/temp_res)) < min_sum; %idx is the index that is NOT an event

Kloten_rain.Event_1(indx == 1) = 1;


for i = ((interevent_dur * 60/temp_res)/2) + 1 : (size(Kloten_rain,1) - (interevent_dur * 60/temp_res)/2)
    if Kloten_rain.Event_1(i) == 0
        Kloten_rain.Event_2(i - (interevent_dur * 60/temp_res)/2:i + (interevent_dur * 60/temp_res)/2) = 1;
    end
end

for i = 2:size(Kloten_rain,1)

    if and(Kloten_rain.Event_2(i) == 1, Kloten_rain.Event_2(i-1) == 0)
        Kloten_rain.Event_3(i) = 1;
    elseif and(Kloten_rain.Event_2(i) == 0, Kloten_rain.Event_2(i-1) == 1)
        Kloten_rain.Event_3(i) = 2;
    end

end

startTime = Kloten_rain.time(Kloten_rain.Event_3==1);
endTime = Kloten_rain.time(Kloten_rain.Event_3==2);


%%
for i = 1:size(endTime,1)
    st_time = startTime(i);
    end_time = endTime(i);

    rain_event{i,1} = Kloten_rain(isbetween(Kloten_rain.time, st_time, end_time),1);
    rain_event{i,2} = mean(rain_event{i,1}.("rain_mm_h-1"));
    rain_event{i,3} = sum(rain_event{i,1}.("rain_mm_h-1")*temp_res/60);
    rain_event{i,6} = unique(year(rain_event{i,1}.time(1)));
    rain_event{i,4} = max(rain_event{i,1}.("rain_mm_h-1"));
    rain_event{i,5} = range(rain_event{i,1}.("rain_mm_h-1"));
    rain_event{i,7} = std(rain_event{i,1}.("rain_mm_h-1"));
    rain_event{i,8} = Kloten_SM(isbetween(Kloten_SM.time, st_time, end_time),1);
    rain_event{i,9} = size(rain_event{i,1},1) * temp_res;
    index = find(~isnan(rain_event{i,8}.soil_moisture), 1, 'first');
    rain_event{i,10} = rain_event{i,8}.soil_moisture(index);
    rain_event{i,11} = find(rain_event{i,1}.("rain_mm_h-1")==max(rain_event{i,1}.("rain_mm_h-1")),1,'first');
    rain_event{i,12} = rain_event{i,11}/(rain_event{i,9});
    rain_event{i,13} = string(st_time);
    rain_event{i,14} = string(end_time);
    rain_event{i,15} = unique(month(rain_event{i,1}.time(1)));

end


%%
rain_mean = cell2table(rain_event(:,2));
rain_mean.Properties.VariableNames{1} = 'mean_mm_h-1';

rain_dep = cell2table(rain_event(:,3));
rain_dep.Properties.VariableNames{1} = 'total_precipitation_mm';

rain_max = cell2table(rain_event(:,4));
rain_max.Properties.VariableNames{1} = 'max_mm_h-1';

rain_range = cell2table(rain_event(:,5));
rain_range.Properties.VariableNames{1} = 'range_mm_h-1';

rain_std = cell2table(rain_event(:,7));
rain_std.Properties.VariableNames{1} = 'std_mm_h-1';

rain_year = cell2table(rain_event(:,6));
rain_year.Properties.VariableNames{1} = 'year';

a(:,1) = 1:size(rain_event,1);
event_no = table(a);
event_no.Properties.VariableNames{1} = 'event_no';

rain_dur = cell2table(rain_event(:,9));
rain_dur.Properties.VariableNames{1} = 'duration_min';

SM_init = table(rain_event(:,10));
SM_init.Properties.VariableNames{1} = 'soil_moisture';

Peak_time = table(rain_event(:,11));
Peak_time.Properties.VariableNames{1} = 'peak_time_min';

Peak_ratio = table(rain_event(:,12));
Peak_ratio.Properties.VariableNames{1} = 'peak_ratio';

st_tt = table(rain_event(:,13));
st_tt.Properties.VariableNames{1} = 'start_time';

end_tt = table(rain_event(:,14));
end_tt.Properties.VariableNames{1} = 'end_time';

rain_month = table(rain_event(:,15));
rain_month.Properties.VariableNames{1} = 'month';

event_table = [rain_mean rain_dep rain_max rain_range rain_std event_no rain_year rain_dur SM_init Peak_time Peak_ratio st_tt end_tt rain_month];

%%
segments_final = [1,1,1,2,3,3,3,3,4,4,1,1];

for mn = 1:size(event_table.duration_min,1)
    for pp = 1:12
        if cell2mat(event_table.month(mn)) == pp
            event_table.segment(mn) = segments_final(pp);
        end
    end
end

%%
figure(5)
cd(fullfile(mainFolder, '_figures\_kloten'))
hold on
text(2.5,50,{'(1): Low rainfall'},'FontSize', 38, 'Rotation',90, 'Color',	'k', 'FontName','Calibri')
xline(4, '--','LineWidth',2, 'HandleVisibility','off', 'Color','#d95f0e')
text(4.5,50,{'(2): Pre-peak rainfall'},'FontSize', 38, 'Rotation',90, 'Color',	'k', 'FontName','Calibri')
xline(5, '--','LineWidth',2, 'HandleVisibility','off', 'Color','#d95f0e')
text(7,50,{'(3): Peak rainfall'},'FontSize', 38, 'Rotation',90, 'Color',	'k', 'FontName','Calibri')
xline(9, '--','LineWidth',2, 'HandleVisibility','off', 'Color','#d95f0e')
xline(11, '--','LineWidth',2, 'HandleVisibility','off', 'Color','#d95f0e')
text(10, 50, {'(4): Post-peak rainfall'},'FontSize', 38, 'Rotation',90,'Color',	'k', 'FontName','Calibri')
text(12,50,{'(1): Low rainfall'},'FontSize', 38, 'Rotation',90, 'Color',	'k', 'FontName','Calibri')
hold off
set(figure(5), 'WindowState', 'maximized');
fontname(gcf,"Calibri")
savefig(gcf,'_kloten\segments_kloten.fig')
exportgraphics(gcf, '_kloten\segments_kloten.jpg', 'Resolution',800)

%%
startYear_1 = 1986;
endYear_1 = 1990;
startYear_2 = 1991;
endYear_2 = 1996;

selectedYearsIndex = (event_table.year >= startYear_1 & event_table.year <= endYear_1) | ...
    (event_table.year >= startYear_2 & event_table.year <= endYear_2);

event_table2 = event_table(selectedYearsIndex,:);

cd(fullfile(mainFolder, '_processed_data\_kloten'))
writetable(event_table2,sprintf('kloten_eventsummary_%s_%d_%d.txt', condition, startYear_1, endYear_2))