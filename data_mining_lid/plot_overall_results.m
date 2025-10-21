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

%%
cd('_processed_data\_kloten\')
cc = 'kloten';
    
addpath('_codes_and_functions')

%%
n_it = 4;

for i = 1:n_it
    early = [];
    late = [];
    if i == 1
        ind_name = 'uf-p_ratio';
        xlab = 'UF-P sum ratio [%]';
        th = 100;
        cl = '#43a2ca';
        alp = 0.65;
        if caseStudy == 2
            binwid = 0.5;
        else
            binwid = 10;
        end

    elseif i == 2
        ind_name = 'centroid';
        if caseStudy == 2
            xlab = 'P-UF centroid time shift [5 min]';
            th = 50;
            binwid = 2;
        else
            xlab = 'P-UF centroid time shift [10 min]';
            th = 10;
            binwid = 0.5;
        end

        cl = '#e6550d';
        alp = 0.35;


    elseif i == 3
        ind_name = 'xcorr';
        if caseStudy == 2
            xlab = 'P-UF correlation time shift [5 min]';
            th = 50;
            binwid = 2;
        else
            xlab = 'P-UF correlation time shift [10 min]';
            th = 10;
            binwid = 0.7;
        end

        cl = '#fec44f';
        alp = 0.35;

    else
        ind_name = 'dtw_median';
        if caseStudy == 2
            xlab = 'P-UF DTW time shift [5 min]';
            th = 50;
            binwid = 2;
        else
            xlab = 'P-UF DTW time shift [10 min]';
            th = 10;
            binwid = 0.8;
        end

        cl = '#0072BD';
        alp = 0.35;


    end

    for j = 1:4
        filename = sprintf('%s_overall_%s_segment%d.txt', cc, ind_name, j);
        if ~exist(filename, 'file')
            fprintf('File %s not found. Skipping j = %d.\n', filename, j);
            continue;
        end
        a = readtable(filename);
        early = vertcat(early, a.Early); % Concatenate first column
        late = vertcat(late, a.Later);   % Concatenate second column
    end


    if i == 1
        ind_zero_flow = or(early == 0, late == 0);
        writetable(table(ind_zero_flow),'ind_zero_flow.txt', WriteVariableNames=false)
    else
        ind_zero_flow = table2array(readtable('ind_zero_flow.txt'));
        ind_zero_flow = logical(ind_zero_flow);
        early = early(~ind_zero_flow);
        late = late(~ind_zero_flow);
    end


    [filtered_early, filtered_late] = remove_outlier_values(early, late);
    [results, required_n] = compare_statistics2(filtered_early, filtered_late, 1000, 0.05, 0.90);
    fprintf('Mean early: %.5f.\n', results.mean_early)
    fprintf('SD early: %.5f).\n', results.std_early)

    fprintf('Mean later: %.5f).\n', results.mean_later)
    fprintf('SD later: %.5f).\n', results.std_later)


    figure(555)
    subplot(2,2,i)
    histogram(filtered_early, 'FaceColor', 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'k', 'BinWidth', binwid)
    hold on
    histogram(filtered_late, 'FaceColor', cl, 'FaceAlpha', alp, 'EdgeColor', cl, 'BinWidth', binwid)
    hold off

    % Add mean lines
    xline(mean(filtered_early), '--', 'color', 'k', 'LineWidth', 2);
    xline(mean(filtered_late), '-.', 'color', cl, 'LineWidth', 2);

    legend('Early', 'Later', 'Mean (Early)', 'Mean (Later)');
    legend(FontSize = 20, FontName ='Calibri')
    xlabel(xlab, 'Interpreter', 'none', 'FontSize', 25, 'FontName', 'Calibri');
    ylabel('Frequency', 'FontSize', 25, 'FontName', 'Calibri');
    set(gca, 'FontName', 'Calibri', 'FontSize', 25)

    title(sprintf('Sample size: %d', size(filtered_early, 1)), 'Interpreter', 'none', 'FontSize', 25, 'FontName', 'Calibri');
    hold off;


end
set(figure(555), 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);

%% Save figure
cd(sprintf('_figures\\_%s', cc));

savefig(gcf, 'all_matches_overview.fig')
exportgraphics(gcf, 'all_matches_overview.png', 'Resolution',1000)

