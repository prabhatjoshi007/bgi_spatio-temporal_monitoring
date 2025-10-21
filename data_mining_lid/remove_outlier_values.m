function [filtered_early, filtered_late] = remove_outlier_values(early, late)
ind_nan = or(isnan(early), isnan(late));
ind_neg = or(early<0, late < 0);
early(ind_nan) = [];
early(ind_neg) = [];
late(ind_nan) = [];
late(ind_neg) = [];


% Compute the 10th and 90th percentiles
ind_early = ~isoutlier(early, 'percentiles', [5 95]);
ind_late = ~isoutlier(late, 'percentiles', [5 95]);
ind_combined = and(ind_early, ind_late);
filtered_early = early(ind_combined);
filtered_late = late(ind_combined);








