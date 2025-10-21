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
%cd('_processed_data\_kloten\')
cc = 'kloten';
cond = 'clogged';
    
addpath('_codes_and_functions')

%%
early = [];
late = [];
ind_name = 'uf-p_ratio';
xlab = 'UF-P sum ratio [%]';
th = 100;
cl = '#43a2ca';
alp = 0.65;
binwid = 10;

%%
n_it = 4;


for j = 1:4
    filename = sprintf('%s_%s_overall_%s_segment%d.txt', cc, cond, ind_name, j);
    if ~exist(filename, 'file')
        fprintf('File %s not found. Skipping j = %d.\n', filename, j);
        continue;
    end
    a = readtable(filename);
    early = vertcat(early, a.early); % Concatenate first column
    late = vertcat(late, a.later);   % Concatenate second column
end
ind_zero_flow = and(early == 0, late == 0);
writetable(table(ind_zero_flow),'ind_zero_flow.txt', WriteVariableNames=false)

filtered_early = early(~ind_zero_flow);
filtered_late = late(~ind_zero_flow);

%[filtered_early, filtered_late] = remove_outlier_values(early, late);
[results, required_n] = compare_statistics2(filtered_early, filtered_late, 1000, 0.05, 0.90);
fprintf('Mean early: %.5f.\n', results.mean_early)
fprintf('SD early: %.5f).\n', results.std_early)

fprintf('Mean later: %.5f).\n', results.mean_later)
fprintf('SD later: %.5f).\n', results.std_later)


figure(555)
plot(1)
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



set(figure(555), 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);

%% Save figure
%cd(sprintf('_figures\\_%s', cc));

savefig(gcf, 'all_matches_overview.fig')
exportgraphics(gcf, 'all_matches_overview.png', 'Resolution',1000)

