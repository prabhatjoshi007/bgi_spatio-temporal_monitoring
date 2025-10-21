function [results, required_n] = compare_statistics2(early, later, n_sim, alpha, threshold)
% estimate_required_sample_size estimates the required sample size per group to
% detect a significant difference between two groups using the Mann–Whitney U test.
%
% USAGE:
%   required_n = estimate_required_sample_size(filtered_early, filtered_late, n_sim, alpha, threshold)
%
% INPUTS:
%   early : Numeric vector of data for the "Early" events.
%   late  : Numeric vector of data for the "Later" events.
%   n_sim          : Number of simulation iterations per candidate sample size.
%   alpha          : Significance level for the test (e.g., 0.05).
%   threshold      : Power threshold to stop simulation (e.g., 0.90 for 90% power).
%
% OUTPUT:
%   required_n     : Estimated required sample size per group to achieve power > threshold.
%                    If no candidate sample size meets the threshold, returns [].
%
% EXAMPLE:
%   filtered_early = [1.2, 2.3, 1.5, 2.0, 1.8, 2.1, 1.9, 2.2];
%   filtered_late  = [2.5, 3.0, 2.7, 3.1, 2.9, 3.2, 3.0, 2.8];
%   n_sim = 1000;
%   alpha = 0.05;
%   threshold = 0.90;
%   required_n = estimate_required_sample_size(filtered_early, filtered_late, n_sim, alpha, threshold);

%% Step 1: Compute Mean and Standard Deviation
mean_early = mean(early);
std_early  = std(early);
mean_later = mean(later);
std_later  = std(later);

% Initialize results structure.
results.mean_early = mean_early;
results.std_early  = std_early;
results.mean_later = mean_later;
results.std_later  = std_later;
results.tStatistic = NaN;   % Default if t-test is not used.
results.testUsed   = '';
results.pValue     = NaN;
results.significant = false;

%% Step 2: Check Normality and Perform the Appropriate Test
% Use alpha_norm as the significance level for the normality tests.
if exist('swtest', 'file') == 2
    % Use Shapiro-Wilk test (if available).
    [h_sw_early, p_sw_early] = swtest(early, alpha);
    [h_sw_later, p_sw_later]   = swtest(later, alpha);
    
    fprintf('\nUsing Shapiro-Wilk test for normality:\n');
    fprintf('  Early: h = %d, p = %.5f\n', h_sw_early, p_sw_early);
    fprintf('  Later: h = %d, p = %.5f\n', h_sw_later, p_sw_later);
    
    isNormal = (~h_sw_early) && (~h_sw_later);
else
    % Fall back to the Lilliefors test if swtest is not available.
    fprintf('\nShapiro-Wilk test not found. Falling back to Lilliefors test.\n');
    [h_norm_early, p_norm_early] = lillietest(early, 'Alpha', alpha);
    [h_norm_later, p_norm_later]   = lillietest(later, 'Alpha', alpha);
    
    fprintf('  Early (Lilliefors): h = %d, p = %.5f\n', h_norm_early, p_norm_early);
    fprintf('  Later (Lilliefors): h = %d, p = %.5f\n', h_norm_later, p_norm_later);
    
    isNormal = (~h_norm_early) && (~h_norm_later);
end

if isNormal
    results.testUsed = 't-test (Welch)';
    % Perform a two-sample t-test (Welch's t-test does not assume equal variances).
    [~, p_ttest, ~, stats] = ttest2(early, later, 'Vartype', 'unequal');
    results.pValue = p_ttest;
    results.tStatistic = stats.tstat;
    
    if p_ttest < 0.05
        results.significant = true;
        fprintf('\nTwo-sample t-test (Welch) indicates a significant difference (p = %.5f).\n', p_ttest);
    else
        fprintf('\nTwo-sample t-test (Welch) indicates no significant difference (p = %.5f).\n', p_ttest);
    end
else
    results.testUsed = 'Mann-Whitney U (ranksum)';
    % Use the non-parametric Mann-Whitney U test.
    p_mwu = ranksum(early, later);
    results.pValue = p_mwu;
    
    if p_mwu < 0.05
        results.significant = true;
        fprintf('\nMann-Whitney U test indicates a significant difference (p = %.5f).\n', p_mwu);
    else
        fprintf('\nMann-Whitney U test indicates no significant difference (p = %.5f).\n', p_mwu);
    end
end




%% Sample size
% Define candidate sample sizes 
sample_sizes = 2:2:n_sim;
power_results = zeros(size(sample_sizes));

fprintf('Simulating power for various sample sizes...\n');

required_n = [];  % Initialize the required sample size variable

% Loop over candidate sample sizes
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx);
    sig_count = 0;

    % Run simulation n_sim times for the current sample size per group
    for sim = 1:n_sim
        % Bootstrap sample (with replacement) from each group
        sample_early = datasample(early, n);
        sample_late  = datasample(later, n);

        % Perform the two-sided Mann–Whitney U test (Wilcoxon rank-sum test)
        p = ranksum(sample_early, sample_late);

        % Count as a "success" if the p-value is below alpha
        if p < alpha
            sig_count = sig_count + 1;
        end
    end

    % Calculate estimated power as the fraction of successful simulations
    power_results(idx) = sig_count / n_sim;
    fprintf('Sample size per group: %3d, Estimated Power: %.3f\n', n, power_results(idx));

    % Stop the simulation once the power exceeds the threshold
    if power_results(idx) >= threshold
        required_n = n;
        break;
    end
end

% Display the result
fprintf('\n==============================================\n');
if ~isempty(required_n)
    fprintf('Estimated required sample size per group to achieve power > %.0f%%: %d\n', threshold*100, required_n);
else
    fprintf('The required sample size is larger than the tested range. Consider increasing the range and rerunning the simulation.\n');
end
fprintf('==============================================\n');
end
