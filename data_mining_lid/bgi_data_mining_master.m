%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Purpose:
% To identify similar rainfall-runoff events across two time periods in different urban catchments ...
% (Oslo, Kloten (period 1), Kloten (period 2)), cluster them using PCA and K-means, and assess changes...
% in hydrological response using DTW, centroid delay, and cross-correlation metrics.

% Main Functionalities:
% - Reads rainfall event summaries and time series for rainfall (P) and runoff (Q).
% - Applies feature normalization and PCA on hydrological characteristics.
% - Uses gap statistic and K-means for clustering events.
% - Identifies similar rainfall events using Dynamic Time Warping (DTW).
% - Evaluates underdrain flow efficiency, centroid delays, and lag using cross-correlation.
% - Visualises temporal changes in hydrological response.
% - Outputs overview data and graphics for reporting.

% Inputs:
% - User input for:
%     -- Case study city (Oslo, Kloten (period 1), Kloten (period 2))
%     -- Box number and segment number
%     -- Whether to write tables or save workspaces
%     -- Number of clusters and plotting/export options
%     -- Event summary text files (*_eventsummary_b*.txt)
%     -- Rainfall and runoff time series files (*_rainfall.txt, *_runoff.txt)

% Dependencies:
% - MATLAB Built-in & Toolboxes
%     -- Statistics and Machine Learning Toolbox (for pca, kmeans, evalclusters, xcorr)
%     -- Signal Processing Toolbox (for movmean, xcorr)
%     -- MATLAB Graphics (for plotting)
%     -- MATLAB Tables (for readtable, writetable, readtimetable)

% Python Integration:
% - Python package: fastdtw

% Custom MATLAB Functions:
% - tip_to_uniform_flow.m
% - rain_similarity_test.m
% - flowchecks.m
% - kge.m
% - centroid.m
% - fastdtw_to_matrix.m
% - grouped_boxchart.m

% Outputs
% - Event tables with cluster indices
% - Clustered event pairs with time-aligned P and UF
% - Text files with underdrain ratios, centroid lags, and cross-correlation lags
% - .fig and .png plots of events and analysis results

% Author & Date:
%   - Prabhat Joshi, prabhat.joshi@eawag.ch, 2025-03-10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clear the command window, workspace, and opened windows
clc
clear
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

%% Add the filepaths and go to the folder of the respective case study
caseStudy = 2;
cd('_processed_data\_kloten\')
addpath('_all_codes_and_functions\')


%%
condition = input('Which scenario to run? 1 --> unclogged; 2 --> clogged: ');

if condition == 1
    cond = 'unclogged';
else
    cond = 'clogged';
end


%% Add python system path to import the fastdtw algorithm

% 1. Import Python’s site module
site = py.importlib.import_module('site');

% 2. Ask Python for your per-user site-packages directory
userSite = char(site.getusersitepackages());

% 3. If it is not already on sys.path, append it (and let Python process any .pth files)
if count(py.sys.path, userSite) == int64(0)
    py.sys.path.append(userSite);
    site.addsitedir(userSite);
end

% Optional extra: also pick up system-wide site-packages
try
    sysSites = cellfun(@char, cell(site.getsitepackages()), 'UniformOutput', false);
    for i = 1:numel(sysSites)
        p = sysSites{i};
        if count(py.sys.path, p) == int64(0)
            py.sys.path.insert(int32(0), p);  % put ahead of other entries
        end
    end
catch
    % getsitepackages() may not be available in some embeddable Python builds
end

% Import the fastdtw algorithm
fastdtw = py.importlib.import_module('fastdtw');

%% Provide the case study and time, and read the data
writeTable = input('Do you want to write the table? 1--> Yes; 2--> No: '); % Choose '1' if running the code for the first time.
boxNumber = 1;

startYear_1 = 1986;
startYear_2 = 1991;
endYear_1 = 1992;
endYear_2 = 1996;
case_name = "kloten";
event_table = readtable(sprintf('%s_eventsummary_%s_%d_%d.txt', case_name, cond, startYear_1, endYear_2));
mult = 1; % converter to have data in mm.h-1 units
t_res = 10; % temporal resolution of data



startYear = startYear_1:startYear_2;
endYear = endYear_1:endYear_2;

%%
Segment_no = input('Which segment to analyse?: ');
event_table = event_table(event_table.segment == Segment_no, :);


%% Perform PCA and determine, among other things, the coefficients, PC scores, number of components to retain

scaled_data = normalize(table2array(event_table(:, {'mean_mm_h_1', 'duration_min', 'total_precipitation_mm', 'max_mm_h_1', 'range_mm_h_1', 'std_mm_h_1', 'peak_time_min', 'peak_ratio'})));

% coeff: Principal component coefficients (loadings)
% score: The principal component scores (projection of the data in the new space)
% latent: Eigenvalues of the covariance matrix (variance explained by each PC)
% tsquared: Hotelling's T-squared statistic
% explained: Percentage of total variance explained by each principal component
% mu: Mean of each feature

rng(1); % This sets the seed to 1
[coeff, score, latent, tsquared, explained, mu] = pca(scaled_data);


% Select the number of principal components and decide how many components to retain based on explained variance
% Chosen value: retain components that explain at least 95% of the variance
cumulativeExplained = cumsum(explained);
numComponents = find(cumulativeExplained >= 95, 1);

fprintf('Number of principal components to retain (to explain at least 95%% of variance): %d\n', numComponents);

% Use the selected number of principal components for further analysis
selectedScores = score(:, 1:numComponents);


%% Use the optimal number of clusters to run K-means clustering again
tic; % Start timer
beforeMem = memory; % Get memory usage before execution
rng(382); % Set the random seed for reproducibility
eval_result = evalclusters(selectedScores, 'kmeans', 'gap', 'KList', 2:10, ReferenceDistribution='uniform');
elapsedTime = toc; % End timer and get elapsed time in seconds
afterMem = memory;
memoryUsed = afterMem.MemUsedMATLAB - beforeMem.MemUsedMATLAB;
fprintf('Elapsed time: %.2f seconds\n', elapsedTime);
fprintf('Memory used: %.2f MB\n', memoryUsed / 1e6);
optimal_k2 = eval_result.OptimalK;
[idx, C] = kmeans(selectedScores, optimal_k2);

% Add cluster index to the table
event_table.cluster = idx;


%% Write the table if desired.
if writeTable == 1
    writetable(event_table, sprintf('%s_event_table_cluster_%s_%d_%d.csv', case_name, cond, startYear_1, endYear_2));
end
%% read the rainfall and flow data for the case studies

allData = event_table;
rain_agg = readtimetable(sprintf('kloten_%s_b%d_rainfall.csv',cond, boxNumber));
flowData = readtimetable(sprintf('kloten_%s_b%d_runoff.csv',cond, boxNumber));

flowData.Properties.VariableNames{1} = 'Qdelta';
flowData.Properties.DimensionNames{1} = 'TIMESTAMP';


rain_agg.Properties.VariableNames{1} = 'Pdelta';
rain_agg.Properties.DimensionNames{1} = 'TIMESTAMP';

%%
timelist = array2table([allData.start_time, allData.end_time]);
timelist.Properties.VariableNames{1} = 'Start_Time';
timelist.Properties.VariableNames{2} = 'End_Time';
timelist.Event_No = allData.event_no;
%%
for i = 1:size(timelist,1)

    st_time = timelist.Start_Time(i);
    %st_time = St(i);

    end_time = timelist.End_Time(i);
    %end_time = Et(i);

    rain_event{i,1} = rain_agg(isbetween(rain_agg.TIMESTAMP, st_time, end_time),1);
    rain_event{i,2} = flowData(isbetween(flowData.TIMESTAMP, st_time, end_time),1);
end
event_no = table(timelist.Event_No);
event_no.Properties.VariableNames{1} = 'EventNo';

event_table2 = [rain_event event_no];%%


%% Select the cluster to analyse
SM_threshold = 0.05; % acceptable difference between the SM of the early and later years
selectedEventTally = 0; % event counter
selectedRainTally = 0; % rain counter
for cluster_no = 1:optimal_k2
    disp('Running cluster no:');
    disp(cluster_no);

    selectedRows = allData(allData.cluster == cluster_no, :);


    % Select the rainfall according to the cluster
    selected_rainfall = event_table2(ismember(event_table2.EventNo, selectedRows.event_no), :);
    selectedRows = selectedRows(ismember(selectedRows.event_no, selected_rainfall.EventNo),:);
    selectedRows.Year_1 = zeros(size(selectedRows,1),1);

    for i = 1:size(selectedRows,1)
        if any(selectedRows.year(i) == startYear_1:startYear_2)
            selectedRows.Year_1(i) = 1;
        else
            selectedRows.Year_1(i) = 0;
        end
    end

    for i = 1:size(selectedRows,1)
        if any(selectedRows.year(i) == endYear_1:endYear_2)
            selectedRows.Year_1(i) = 2;
        end
    end

    % Perform DTW within the rainfall events within a cluster to shortlist "similar" events
    similarRain = zeros(size(selected_rainfall,1), size(selected_rainfall,1));
    for i = 1:size(selected_rainfall,1)
        for j = 1:size(selected_rainfall,1)

            if i>j
                similarRain(i,j) = 1e6;
            elseif i==j
                similarRain(i,j) = 1e6;
            else
                distance = fastdtw.fastdtw(wd_dtw_input(selected_rainfall.Var1{i,1}.Pdelta), wd_dtw_input(selected_rainfall.Var1{j,1}.Pdelta));
                similarRain(i,j) = double(distance(1));
            end
        end
    end

    % Select the five most similar rainfall events
    [minValues, linearIndices] = sort(similarRain(:));

    % Select the first 5 minimum values and their positions
    minValues = minValues(1:size(similarRain,1)*(size(similarRain,1)-1)/2);
    linearIndices = linearIndices(1:size(similarRain,1)*(size(similarRain,1)-1)/2);

    % Convert linear indices to row and column indices
    [rowIndices, colIndices] = ind2sub(size(similarRain), linearIndices);


    mm = length(rowIndices);

    for k = 1:mm
        disp('Running event: ')
        disp(k)
        rainEvent_1 = selected_rainfall.EventNo(rowIndices(k));
        rainEvent_2 = selected_rainfall.EventNo(colIndices(k));


        event_1 = selected_rainfall.Var1(selected_rainfall.EventNo == rainEvent_1);
        event_1{1,2} = selectedRows.Year_1(selected_rainfall.EventNo == rainEvent_1);
        event_1{1,3} = selectedRows.soil_moisture(selected_rainfall.EventNo == rainEvent_1);
        event_1{1,4} = rainEvent_1;

        event_2 = selected_rainfall.Var1(selected_rainfall.EventNo == rainEvent_2);
        event_2{1,2} = selectedRows.Year_1(selected_rainfall.EventNo == rainEvent_2);
        event_2{1,3} = selectedRows.soil_moisture(selected_rainfall.EventNo == rainEvent_2);
        event_2{1,4} = rainEvent_2;

        event_3 = selected_rainfall.Var2(selected_rainfall.EventNo == rainEvent_1);
        event_4 = selected_rainfall.Var2(selected_rainfall.EventNo == rainEvent_2);


        event_1{1,1}.Pdelta(isnan(event_1{1,1}.Pdelta)) = 0;
        event_1{1,1}.Pdelta = tip_to_uniform_flow(event_1{1,1}.Pdelta);

        event_2{1,1}.Pdelta(isnan(event_2{1,1}.Pdelta)) = 0;
        event_2{1,1}.Pdelta = tip_to_uniform_flow(event_2{1,1}.Pdelta);

        event_3{1,1}.Qdelta(isnan(event_3{1,1}.Qdelta)) = 0;
        if sum(event_3{1,1}.Qdelta) ~= 0
            event_3{1,1}.Qdelta = tip_to_uniform_flow(event_3{1,1}.Qdelta);
        end

        event_4{1,1}.Qdelta(isnan(event_4{1,1}.Qdelta)) = 0;
        if sum(event_4{1,1}.Qdelta) ~= 0
            event_4{1,1}.Qdelta = tip_to_uniform_flow(event_4{1,1}.Qdelta);
        end

        min_l = min(size(event_3{1,1},1), size(event_4{1,1},1));

        if or(size(event_1{1,1}.Pdelta(1:min_l),1) < 3, size(event_2{1,1}.Pdelta(1:min_l),1) < 3)
            disp('rain too small...')
            similarRainTest = false;
        else
            similarRainTest = rain_similarity_test(event_1{1,1}.Pdelta(1:min_l), event_2{1,1}.Pdelta(1:min_l), event_1{1,2}, event_2{1,2});
        end


        if similarRainTest == true
            selectedRainTally = selectedRainTally + 1;
            flowconditionsMet = flowchecks(event_1{1,1}.Pdelta(1:min_l), event_2{1,1}.Pdelta(1:min_l), ...
                event_3{1,1}.Qdelta(1:min_l), event_4{1,1}.Qdelta(1:min_l), event_1{1,2}, event_2{1,2}, ...
                event_1{1,3}, event_2{1,3}, SM_threshold);
        else
            flowconditionsMet = false;
        end

        if flowconditionsMet
            disp('Flow conditions met.')
            selectedEventTally = selectedEventTally + 1;


            selectedEventsAll{boxNumber, selectedEventTally} = [event_1{1,1}.Pdelta(1:min_l) event_2{1,1}.Pdelta(1:min_l) event_3{1,1}.Qdelta(1:min_l) event_4{1,1}.Qdelta(1:min_l)];
            yearSelected{boxNumber, selectedEventTally} = [unique(event_1{1,1}.TIMESTAMP(1)) unique(event_2{1,1}.TIMESTAMP(1))];
            monthSelected{boxNumber, selectedEventTally} = [unique(month(event_1{1,1}.TIMESTAMP(1))) unique(month(event_2{1,1}.TIMESTAMP(1)))];
            SMSelected{boxNumber, selectedEventTally} = [string(event_1{1,3}) string(event_2{1,3})];

            event_no_selected{boxNumber, selectedEventTally} = [event_1{1,4} event_2{1,4}];

        end
    end
end


%% Overall picture
for lll = 1:selectedEventTally
    underdrain_all_1{boxNumber, lll} = (sum(selectedEventsAll{boxNumber,lll}(:,3)* mult * (t_res/60)))/(sum(selectedEventsAll{boxNumber,lll}(:,1)* mult * (t_res/60))) * 100 ;
    underdrain_all_2{boxNumber, lll} = (sum(selectedEventsAll{boxNumber,lll}(:,4)* mult * (t_res/60)))/(sum(selectedEventsAll{boxNumber,lll}(:,2)* mult * (t_res/60))) * 100 ;
    underdrain_allresults(lll,:) = [underdrain_all_1{boxNumber, lll} underdrain_all_2{boxNumber, lll}];


    centroid_all_1{boxNumber, lll} = centroid(selectedEventsAll{boxNumber,lll}(:,3)) - centroid(selectedEventsAll{boxNumber,lll}(:,1));

    if centroid_all_1{boxNumber, lll} < 1
        centroid_all_1{boxNumber, lll} = 0;
    end

    centroid_all_2{boxNumber, lll} = centroid(selectedEventsAll{boxNumber,lll}(:,4)) - centroid(selectedEventsAll{boxNumber,lll}(:,2));
    if centroid_all_2{boxNumber, lll} < 1
        centroid_all_2{boxNumber, lll} = 0;
    end

    centroid_allresults(lll,:) = [centroid_all_1{boxNumber, lll} centroid_all_2{boxNumber, lll}];


    [correlation_all_1, lags_all_1] = xcorr(selectedEventsAll{boxNumber,lll}(:,3), selectedEventsAll{boxNumber,lll}(:,1));
    [~, maxIndex_all_1] = max(correlation_all_1);
    lagAtMaxCorrelation_all_1{boxNumber, lll} = lags_all_1(maxIndex_all_1);


    [correlation_all_2, lags_all_2] = xcorr(selectedEventsAll{boxNumber,lll}(:,4), selectedEventsAll{boxNumber,lll}(:,2));
    [~, maxIndex_all_2] = max(correlation_all_2);
    lagAtMaxCorrelation_all_2{boxNumber, lll} = lags_all_2(maxIndex_all_2);



    correlation_allresults(lll,:) = [lagAtMaxCorrelation_all_1{boxNumber, lll} lagAtMaxCorrelation_all_2{boxNumber, lll}];


    rng(445); % Set the random seed for reproducibility
    before_fastdtw = fastdtw.fastdtw(wd_dtw_input(selectedEventsAll{boxNumber,lll}(:,1)), wd_dtw_input(selectedEventsAll{boxNumber,lll}(:,3)));
    before_fastdtw_path = fastdtw_to_matrix(before_fastdtw);

    after_fastdtw = fastdtw.fastdtw(wd_dtw_input(selectedEventsAll{boxNumber,lll}(:,2)), wd_dtw_input(selectedEventsAll{boxNumber,lll}(:,4)));
    after_fastdtw_path = fastdtw_to_matrix(after_fastdtw);

    s1 = before_fastdtw_path(:,2) - before_fastdtw_path(:,1);
    s2 = after_fastdtw_path(:,2) - after_fastdtw_path(:,1);

    dtw_allresults(lll,:) = [median(s1) median(s2)];

end

writetable(array2table(underdrain_allresults, 'VariableNames', {'early', 'later'}), sprintf('%s_%s_overall_uf-p_ratio_segment%d.txt', case_name, cond, Segment_no))
writetable(array2table(centroid_allresults, 'VariableNames', {'early', 'later'}), sprintf('%s_%s_overall_centroid_segment%d.txt', case_name, cond, Segment_no))
writetable(array2table(correlation_allresults, 'VariableNames', {'early', 'later'}), sprintf('%s_%s_overall_xcorr_segment%d.txt', case_name, cond, Segment_no))
writetable(array2table(dtw_allresults, 'VariableNames', {'early', 'later'}), sprintf('%s_%s_overall_dtw_median_segment%d.txt', case_name, cond, Segment_no))


%%
% Plot the figures
if caseStudy == 1
    xlab = 'Time [5 min]';
else
    xlab = 'Time [10 min]';
end

if selectedEventTally > 10
    n_subplots = 10;
else
    n_subplots = selectedEventTally;
end

for cc = 1:n_subplots

    centroid_1{boxNumber, cc} = centroid(selectedEventsAll{boxNumber,cc}(:,3)) - centroid(selectedEventsAll{boxNumber,cc}(:,1));
    centroid_2{boxNumber, cc} = centroid(selectedEventsAll{boxNumber,cc}(:,4)) - centroid(selectedEventsAll{boxNumber,cc}(:,2));

    centroid_all = [centroid_1{boxNumber, cc} centroid_2{boxNumber, cc}];

    [correlation_1, lags_1] = xcorr(selectedEventsAll{boxNumber,cc}(:,3), selectedEventsAll{boxNumber,cc}(:,1));
    [~, maxIndex_1] = max(correlation_1);
    lagAtMaxCorrelation_1{boxNumber, cc} = lags_1(maxIndex_1);

    [correlation_2, lags_2] = xcorr(selectedEventsAll{boxNumber,cc}(:,4), selectedEventsAll{boxNumber,cc}(:,2));
    [~, maxIndex_2] = max(correlation_2);
    lagAtMaxCorrelation_2{boxNumber, cc} = lags_2(maxIndex_2);

    correlation_all = [lagAtMaxCorrelation_1{boxNumber, cc} lagAtMaxCorrelation_2{boxNumber, cc}];

    kge_rain = kge(selectedEventsAll{boxNumber,cc}(:,1) * mult, selectedEventsAll{boxNumber,cc}(:,2) * mult);
    kge_uf = kge(selectedEventsAll{boxNumber,cc}(:,3) * mult, selectedEventsAll{boxNumber,cc}(:,4) * mult);
    kge_string = {sprintf('KGE_P: %0.2f', kge_rain), sprintf('KGE_{UF}: %0.2f', kge_uf)};

    figure(999)
    subplot(5, 2, cc)
    plot(selectedEventsAll{boxNumber,cc}(:,1) * mult, '-', 'LineWidth', 1.5, 'Color','#999999', ...
        DisplayName = 'P; Early; SM: ' + SMSelected{boxNumber,cc}(1)');
    hold on
    plot(selectedEventsAll{boxNumber,cc}(:,2) * mult, '--', 'LineWidth', 1.5, 'Color','k', ...
        DisplayName = 'P; Later; SM: ' + SMSelected{boxNumber,cc}(2)') ;
    ylabel('P [mm.h-1]');
    ylim([0 10])
    set(gca, 'YColor', 'k'); % Set the color of the left y-axis to blue
    set(gca, 'FontSize',8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    text(0.72 * size(selectedEventsAll{boxNumber,cc}(:,4),1), 2, kge_string,'FontSize',8)
    legend('FontSize',8, 'Location','northwest','Color','none', 'EdgeColor','none')
    fontname(gcf,"Calibri")
    set(figure(999), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;

    figure(998)
    subplot(5, 2, cc)
    hold on
    plot(cumsum(selectedEventsAll{boxNumber,cc}(:,1)) * mult * (t_res/60), '-', 'LineWidth', 1.5, 'Color','#999999', ...
        DisplayName = 'P; Early; SM: ' + string(SMSelected{boxNumber,cc}(1)));
    plot(cumsum(selectedEventsAll{boxNumber,cc}(:,2)) * mult * (t_res/60), '--', 'LineWidth', 1.5, 'Color','k', ...
        DisplayName = 'P; Later; SM: ' + string(SMSelected{boxNumber,cc}(2)));
    hold off
    grid on
    xlabel(xlab);
    ylabel('Sum [mm]');
    ylim([0 8])
    set(gca, 'FontSize',8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    legend('FontSize',8, 'Location','northwest', 'Color','none', 'EdgeColor','none')
    fontname(gcf,"Calibri")
    set(figure(998), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;

    figure(100)
    subplot(5, 2, cc)
    yyaxis left;
    plot(selectedEventsAll{boxNumber,cc}(:,3) * mult, '-', 'LineWidth', 1.5, 'Color','#43a2ca', ...
        DisplayName = 'UF; Early; SM: ' + SMSelected{boxNumber,cc}(1));
    hold on
    plot(selectedEventsAll{boxNumber,cc}(:,4) * mult, '-', 'LineWidth', 1.5, 'Color','#2ca25f', ...
        DisplayName = 'UF; Later; SM: ' + SMSelected{boxNumber,cc}(2));
    ylim([0 5])
    ylabel('UF [mm.h-1]');
    xlabel(xlab);
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    set(gca, 'YColor', 'k'); % Set the color of the left y-axis to blue
    set(gca, 'FontSize', 8)
    text(0.72 * size(selectedEventsAll{boxNumber,cc}(:,4),1), 2, kge_string,'FontSize',8)
    grid on;


    % Plot the rainfall on the right y-axis with reversed axis
    yyaxis right;
    plot(selectedEventsAll{boxNumber,cc}(:,1) * mult, '-', 'LineWidth', 1.5, 'Color','#999999', ...
        DisplayName = 'P; Early');
    hold on
    plot(selectedEventsAll{boxNumber,cc}(:,2) * mult, '--', 'LineWidth', 1.5, 'Color','k', ...
        DisplayName = 'P; Later') ;
    ylabel('P [mm.h-1]');
    ylim([0 5])
    set(gca, 'YColor', 'k'); % Set the color of the left y-axis to blue
    set(gca, 'YDir', 'reverse'); % Reverse the y-axis
    set(gca, 'FontSize',8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    legend('FontSize',8, 'Location','southwest','Color','none', 'EdgeColor','none')
    fontname(gcf,"Calibri")
    set(figure(100), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);

    %
    figure(101)
    subplot(5, 2, cc)
    plot(cumsum(selectedEventsAll{boxNumber,cc}(:,3)) * mult * (t_res/60), '-', 'LineWidth', 1.5, 'Color','#43a2ca', ...
        DisplayName = 'UF; Early; SM: ' + string(SMSelected{boxNumber,cc}(1)));
    hold on
    plot(cumsum(selectedEventsAll{boxNumber,cc}(:,4)) * mult * (t_res/60), '-', 'LineWidth', 1.5, 'Color','#2ca25f', ...
        DisplayName = 'UF; Later; SM: ' + string(SMSelected{boxNumber,cc}(2)));
    plot(cumsum(selectedEventsAll{boxNumber,cc}(:,1)) * mult * (t_res/60), '-', 'LineWidth', 1.5, 'Color','#999999', ...
        DisplayName = 'P; Early');
    plot(cumsum(selectedEventsAll{boxNumber,cc}(:,2)) * mult * (t_res/60), '--', 'LineWidth', 1.5, 'Color','k', ...
        DisplayName = 'P; Later');
    hold off
    grid on
    xlabel(xlab);
    ylabel('Sum [mm]');
    ylim([0 60])
    set(gca, 'FontSize',8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    %xlim([0 120])
    legend('FontSize',8, 'Location','northwest', 'Color','none', 'EdgeColor','none')
    fontname(gcf,"Calibri")
    set(figure(101), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);


    rng(445); % Set the random seed for reproducibility

    before_fastdtw = fastdtw.fastdtw(wd_dtw_input(selectedEventsAll{boxNumber,cc}(:,1)), wd_dtw_input(selectedEventsAll{boxNumber,cc}(:,3)));
    before_fastdtw_path = fastdtw_to_matrix(before_fastdtw);

    after_fastdtw = fastdtw.fastdtw(wd_dtw_input(selectedEventsAll{boxNumber,cc}(:,2)), wd_dtw_input(selectedEventsAll{boxNumber,cc}(:,4)));
    after_fastdtw_path = fastdtw_to_matrix(after_fastdtw);

    s1 = before_fastdtw_path(:,2) - before_fastdtw_path(:,1);
    s1 = vertcat(s1, centroid_all(1));
    s1 = vertcat(s1, correlation_all(1));

    s2 = after_fastdtw_path(:,2) - after_fastdtw_path(:,1);
    s2 = vertcat(s2, centroid_all(2));
    s2 = vertcat(s2, correlation_all(2));

    s = [s1; s2];

    g1 = repmat('Early',length(before_fastdtw_path)+2,1);
    g2 = repmat('Later',length(after_fastdtw_path)+2,1);
    g = [g1; g2];

    h1 = repmat('DTW',length(before_fastdtw_path),1);
    h1 = vertcat(categorical(string(h1)), 'Centroid');
    h1 = vertcat(h1, 'Cross-correlation');

    h2 = repmat('DTW',length(after_fastdtw_path),1);
    h2 = vertcat(categorical(string(h2)), 'Centroid');
    h2 = vertcat(h2, 'Cross-correlation');

    h = [h1;h2];


    figure(103)
    subplot(5, 2, cc)
    boxchart(categorical(string(g)), s, 'Notch','off', 'GroupByColor', h)
    ylabel(sprintf('Time shift [%d min] \n between P and UF', t_res))
    set(gca, 'Fontsize', 8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    fontname(gcf,"Calibri")
    set(figure(103), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;
    ylim([-10 20])

    if cc == 1
        legend({'DTW', 'Centroid', 'Cross-correlation'}, 'Location', 'NorthEast');
    else
        legend('off')
    end

    underdrain_1{boxNumber, cc} = (sum(selectedEventsAll{boxNumber,cc}(:,3)* mult * (t_res/60)))/(sum(selectedEventsAll{boxNumber,cc}(:,1)* mult * (t_res/60))) * 100 ;
    underdrain_2{boxNumber, cc} = (sum(selectedEventsAll{boxNumber,cc}(:,4)* mult * (t_res/60)))/(sum(selectedEventsAll{boxNumber,cc}(:,2)* mult * (t_res/60))) * 100 ;
    underdrain = [underdrain_1{boxNumber, cc} underdrain_2{boxNumber, cc}];

    figure(102)
    subplot(5, 2, cc)
    b1 = bar(categorical({'Early', 'Later'}),underdrain,'FaceColor','#43a2ca', 'EdgeColor', '#43a2ca', 'BarWidth', 0.5);
    ylabel('UF-P ratio [%]')
    set(gca, 'Fontsize', 8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    ylim([0 100])
    fontname(gcf,"Calibri")
    set(figure(102), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;


    figure(104)
    subplot(5, 2, cc)
    b2 = bar(categorical({'Early', 'Later'}), correlation_all, 'FaceColor','#80cdc1');

    ylabel(sprintf('Time lags [%d min] from \n cross-correlation \n between P and UF', t_res))
    set(gca, 'Fontsize', 8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    ylim([0 10])
    fontname(gcf,"Calibri")
    set(figure(104), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;


    figure(105)
    subplot(5, 2, cc)
    b3 = bar(categorical({'Early', 'Later'}), centroid_all, 'FaceColor','#fa9fb5');
    ylabel(sprintf('Centroid delay [%d min] \n between P and UF', t_res))
    set(gca, 'Fontsize', 8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)

    ylim([0 10])

    fontname(gcf,"Calibri")
    set(figure(105), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;


    figure(106)
    subplot(5, 2, cc)
    b4 = bar(categorical({'Early', 'Later'}), [centroid_all(1) correlation_all(1); centroid_all(2) correlation_all(2)]);
    b4(1).FaceColor = '#fdcc8a';
    b4(1).EdgeColor = '#fdcc8a';
    b4(2).FaceColor = '#fc8d59';
    b4(2).EdgeColor = '#fc8d59';
    b4(2).FaceAlpha = 1;
    ylabel(sprintf('Time delay [%d min] \n between P and UF', t_res))
    set(gca, 'Fontsize', 8)
    title(sprintf('Early: %s; Later: %s', datestr(yearSelected{boxNumber,cc}(1), 'yyyy-mm-dd hh:MM'), datestr(yearSelected{boxNumber,cc}(2), 'yyyy-mm-dd hh:MM')), 'FontSize',7.5)
    %xlim([0 120])
    ylim([0 10])

    if cc == 1
        legend({'Centroid', 'Cross-correlation'}, 'Location', 'NorthEast');
    else
        legend('off')
    end
    fontname(gcf,"Calibri")
    set(figure(106), 'Units', 'centimeters', 'Position', [2, 2, 18, 25.7]);
    grid on;

end