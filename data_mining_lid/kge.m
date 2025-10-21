%--------------------------------------------------------------------------
% Function: kge
%
% Description:
%   Computes the Kling-Gupta Efficiency (KGE) metric to assess the 
%   performance of a hydrological model by comparing simulated and 
%   observed time series data.
%
%   The KGE metric integrates three components:
%     - Correlation (r) – measures linear agreement
%     - Bias ratio (β) – ratio of means (mean(sim)/mean(obs))
%     - Variability ratio (γ) – ratio of coefficients of variation
%
%   KGE ranges from -∞ to 1, where:
%     - 1 indicates perfect agreement,
%     - 0 indicates performance no better than the mean of observed,
%     - Negative values indicate poor performance.
%
% Syntax:
%   kge_value = kge(TS1, TS2)
%
% Inputs:
%   TS1 - A vector/time series of a variable (usually simulated values from
%   a model)
%   observed  - A vector/time series of a variable (usually corresponding
%   observed values)
%
% Output:
%   kge_value - The Kling-Gupta Efficiency (scalar)
%
% Notes:
%   - NaN values are automatically excluded from analysis.
%   - Inputs are reshaped to column vectors internally.
%   - This implementation uses the original KGE formulation (Gupta et al., 2009).
%
% Reference:
%   Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
%   Decomposition of the mean squared error and NSE performance criteria: 
%   Implications for improving hydrological modelling. 
%   Journal of Hydrology, 377(1–2), 80–91.
%
% Example:
%   kge_value = kge(TS1, TS2);
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------



function kge_value = kge(simulated, observed)

% Ensure inputs are column vectors
simulated = simulated(:);
observed = observed(:);

% Remove NaN values
validIdx = ~isnan(simulated) & ~isnan(observed);
simulated = simulated(validIdx);
observed = observed(validIdx);

% Calculate the components of KGE
% 1. Correlation coefficient (r)
r = corr(simulated, observed);

% 2. Bias ratio (beta)
beta = mean(simulated) / mean(observed);

% 3. Variability ratio (gamma)
gamma = (std(simulated) / mean(simulated)) / (std(observed) / mean(observed));

% KGE calculation
kge_value = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2);
end
