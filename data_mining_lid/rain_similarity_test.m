%--------------------------------------------------------------------------
% Function: rain_similarity_test
%
% Description:
%   Evaluates whether two rainfall time series (e.g., storm events) are 
%   statistically similar based on a combination of tests:
%     1. Kolmogorov–Smirnov test (distribution similarity)
%     2. Normality tests (Shapiro-Wilk)
%     3. Parametric (t-test, F-test) or non-parametric (Mann-Whitney U, Brown-Forsythe) 
%        tests for central tendency and variance
%     4. Kling-Gupta Efficiency (KGE) for hydrological agreement
%
%   This function is useful for pairing rainfall events in different years
%   for comparative hydrological modelling or paired analysis.
%
% Syntax:
%   similarRainTest = rain_similarity_test(rainfall_2011, rainfall_2015, year1, year2)
%
% Inputs:
%   rainfall_2011 - Numeric vector of rainfall data for year1 event
%   rainfall_2015 - Numeric vector of rainfall data for year2 event
%   year1         - Scalar indicating the year for rainfall_2011
%   year2         - Scalar indicating the year for rainfall_2015
%
% Output:
%   similarRainTest - Logical (true/false). Returns true only if:
%     - The events are from different years
%     - Their distributions (via KS test) are not significantly different
%     - Their means/medians are statistically similar (t-test or Mann-Whitney)
%     - Their variances are not significantly different (F-test or Brown-Forsythe)
%     - The Kling-Gupta Efficiency (KGE) is ≥ 0.60
%
% Notes:
%   - This function assumes that rainfall time series are aligned in time and length.
%   - `swtest` must be available in the path for Shapiro-Wilk normality testing.
%   - Requires the user-defined function `kge()` to be accessible.
%   - Outputs verbose reasoning for each rejection condition.
%
% Example:
%   isSimilar = rain_similarity_test(stormA, stormB, 2011, 2015);
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------



function similarRainTest = rain_similarity_test(rainfall_2011, rainfall_2015, year1, year2)

if year1 == year2
    similarRainTest = false;
    fprintf('Same year condition met.\n')
    return
end

% 1. Check if samples come from the same distribution
[h_ks, p_ks] = kstest2(rainfall_2011, rainfall_2015);

kge_value = kge(rainfall_2011, rainfall_2015);


% 2. Check normality of samples
[~, p1_norm] = swtest(rainfall_2011);
[~, p2_norm] = swtest(rainfall_2015);

if p1_norm > 0.05 && p2_norm > 0.05
    % Both samples are normally distributed
    % 3. Compare means using two-sample t-test
    [h_t, p_t] = ttest2(rainfall_2011, rainfall_2015);
    [h_f, p_f] = vartest2(rainfall_2011, rainfall_2015);
else
    % At least one sample is not normally distributed
    % 3. Compare medians using Mann-Whitney U test
    [p_mw, h_mw] = ranksum(rainfall_2011, rainfall_2015);
    [p_bf, ~] = vartestn([rainfall_2011, rainfall_2015], 'TestType','BrownForsythe','Display','off');

end

% 4. Compare variances using F-test


% Display results
%fprintf('Kolmogorov-Smirnov Test p-value: %f\n', p_ks);
if h_ks == 1
    fprintf('The null hypothesis (same distribution) is rejected.\n');
    similarRainTest = false;
    return
else
    fprintf('The null hypothesis (same distribution) cannot be rejected.\n');
    fprintf('Kolmogorov-Smirnov Test p-value: %f\n', p_ks)
end

if p1_norm > 0.05 && p2_norm > 0.05
    fprintf('Both samples are normally distributed.\n');
    fprintf('Two-sample t-test p-value: %f\n', p_t);
    if h_t == 1
        fprintf('The null hypothesis (means are the same) is rejected.\n');
        similarRainTest = false;
        return
    else
        fprintf('The null hypothesis (means are the same) can not be rejected.\n');
    end

    fprintf('F-Test p-value: %f\n', p_f);
    if h_f == 1
        fprintf('The null hypothesis (variances are the same) is rejected.\n');
        similarRainTest = false;
        return
    else
        fprintf('The null hypothesis (variances are the same) can not be rejected.\n');
    end

else
    fprintf('At least one sample is not normally distributed.\n');
    fprintf('Mann-Whitney U Test () p-value: %f\n', p_mw);
    if h_mw == 1
        fprintf('The null hypothesis (medians are the same) is rejected.\n');
        similarRainTest = false;
        return
    else
        fprintf('The null hypothesis (medians are the same) can not be rejected.\n');
    end

    fprintf('BF-Test p-value: %f\n', p_bf);
    if p_bf < 0.05
        fprintf('The null hypothesis (variances are the same) is rejected.\n');
        similarRainTest = false;
        return
    else
        fprintf('The null hypothesis (variances are the same) can not be rejected.\n');
    end
end


if kge_value < 0.60
    fprintf('The KGE value is less than 0.60.\n')
    similarRainTest = false;
    return
else
    fprintf('The KGE criteria is met.\n')
end

similarRainTest = true;
end
