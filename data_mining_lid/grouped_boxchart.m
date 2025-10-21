%--------------------------------------------------------------------------
% Function: groupedBoxchart
%
% Description:
%   Creates a grouped boxplot for visual comparison of subgroups within 
%   multiple primary groups. Each group may contain multiple subgroups 
%   (e.g., 'Early' vs. 'Later') with individual colours and outlier markers.
%   This function supports custom labelling and automatic annotations of 
%   sample sizes, with visual enhancements such as group dividers and legends.
%
% Syntax:
%   groupedBoxchart(data, numGroups, subgroupLabels, groupLabels, ...
%                   yLabelText, subgroupColors, outlierColors)
%
% Inputs:
%   data           - A cell array of size [numGroups x numSubgroups], 
%                    each cell containing a numeric vector of observations.
%   numGroups      - Integer specifying the number of primary groups.
%   subgroupLabels - Cell array of labels for each subgroup 
%                    (e.g., {'Early', 'Later'}).
%   groupLabels    - Cell array of labels for each primary group 
%                    (e.g., {'Zone 1', 'Zone 2'}).
%   yLabelText     - String for Y-axis label (e.g., 'Peak Flow (m³/s)').
%   subgroupColors - Cell array of HEX color codes for each subgroup 
%                    (e.g., {'#f7fcb9', '#addd8e'}).
%   outlierColors  - Cell array of HEX color codes for outliers 
%                    (e.g., {'#ff0000', '#0000ff'}).
%
% Output:
%   A grouped box chart is plotted to the current figure window, with:
%     - Colour-coded boxes per subgroup
%     - Group-wise n-count annotations
%     - Dashed vertical dividers between groups
%     - Custom X and Y labels
%     - Subgroup legend
%
% Notes:
%   - Assumes uniform number of subgroups per group.
%   - Automatically converts HEX colours to RGB for plotting.
%   - Text annotation assumes 2 subgroups for n calculation (adjust if different).
%
% Example:
%   groupedBoxchart(myData, 3, {'Early', 'Later'}, {'Zone A', 'Zone B', 'Zone C'}, ...
%                   'Runoff Volume (m³)', {'#1f77b4','#ff7f0e'}, {'#d62728','#9467bd'});
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------


function groupedBoxchart(data, numGroups, subgroupLabels, groupLabels, yLabelText, subgroupColors, outlierColors)
% groupedBoxchart Creates grouped box charts for data with subgroups
%
% INPUTS:
%   - data: A cell array where each cell contains data for a subgroup.
%           The size should be [numGroups x numSubgroups].
%   - numGroups: The number of primary groups.
%   - subgroupLabels: Cell array of labels for the subgroups (e.g., {'Early', 'Later'}).
%   - groupLabels: Cell array of labels for the groups (e.g., {'Group 1', 'Group 2', ...}).
%   - yLabelText: Label for the Y-axis.
%   - subgroupColors: Cell array of HEX color codes for the subgroups (e.g., {'#f7fcb9', '#addd8e'}).
%   - outlierColors: Cell array of HEX color codes for outliers (e.g., {'#ff0000', '#0000ff'}).

% Validate input sizes
[numGroupsData, numSubgroups] = size(data);
if numGroups ~= numGroupsData
    error('The number of groups in the data does not match the specified numGroups.');
end
if length(subgroupLabels) ~= numSubgroups
    error('The length of subgroupLabels must match the number of subgroups in the data.');
end
if length(subgroupColors) ~= numSubgroups
    error('The length of subgroupColors must match the number of subgroups in the data.');
end
if length(outlierColors) ~= numSubgroups
    error('The length of outlierColors must match the number of subgroups in the data.');
end

% Flatten data into vectors for boxchart
flattenedData = [];
groupIds = [];
subgroupIds = [];
dataCounts = zeros(numGroups, 1); % Track total counts for each group
for g = 1:numGroups
    for s = 1:numSubgroups
        % Add data and identifiers
        flattenedData = [flattenedData; data{g, s}];
        groupIds = [groupIds; g * ones(size(data{g, s}))];
        subgroupIds = [subgroupIds; s * ones(size(data{g, s}))];
    end
    % Update total count for the group
    dataCounts(g) = dataCounts(g) + sum(groupIds == g);
end

% Convert HEX colors to RGB
colorMap = zeros(length(subgroupColors), 3);
for i = 1:length(subgroupColors)
    colorMap(i, :) = sscanf(subgroupColors{i}(2:end), '%2x%2x%2x', [1, 3]) / 255;
end

% Convert outlier HEX color to RGB
% Convert outlier HEX colors to RGB
outlierMap = zeros(length(outlierColors), 3);
for i = 1:length(outlierColors)
    outlierMap(i, :) = sscanf(outlierColors{i}(2:end), '%2x%2x%2x', [1, 3]) / 255;
end

% Create the box chart
figure();
hold on;
for s = 1:numSubgroups
    % Plot box charts for each subgroup
    idx = (subgroupIds == s);
    boxchart(groupIds(idx) + (s - 1) * 0.2, flattenedData(idx), ...
        'BoxFaceColor', colorMap(s, :), 'BoxWidth', 0.15, 'MarkerColor', outlierMap(s, :));
end
hold off;

% Add text annotations for data counts
for g = 1:numGroups
    text(g, 0.97 * range(ylim), sprintf('n=%d', dataCounts(g)/2), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 15);
end
hold off;

% Add vertical lines to separate groups
for g = 1:numGroups - 1
    xLinePos = g + 0.5; % Position of the vertical line at the group boundary
    line([xLinePos, xLinePos], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
end

% Adjust x-axis for group labels
xticks(1:numGroups);
xticklabels(groupLabels);

% Add labels and title
xlabel('Segments');
ylabel(yLabelText);

% Create legend manually

legend('Early', 'Later', 'Location','northeast');

fontname(gcf, 'Calibri');
set(gca, 'FontSize', 15);
grid on;
end
