%--------------------------------------------------------------------------
% Function: wd_dtw_input
%
% Description:
%   Prepares a time series for Weighted Derivative Dynamic Time Warping (WD-DTW)
%   by adding small Gaussian noise to reduce potential alignment ties, and 
%   computing the first-order difference of the resulting series.
%
%   This is useful in hydrology, signal processing, or any time series domain 
%   where derivative-based DTW improves matching of temporal trends rather 
%   than absolute values.
%
% Syntax:
%   diff_data = wd_dtw_input(data)
%
% Input:
%   data - A numeric vector (time series signal) to be processed.
%
% Output:
%   diff_data - A vector containing the first-order difference of the 
%               noise-perturbed signal. Length is `length(data) - 1`.
%
% Notes:
%   - Noise added is Gaussian with σ = 1% of the mean of `data`.
%   - This can help improve DTW matching by breaking ties in flat signals 
%     and mimicking subtle variability.
%
% Example:
%   raw = [1, 2, 2, 3, 4];
%   x = wd_dtw_input(raw);
%   plot(x); title('Derivative input for WD-DTW');
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------


function [diff_data] = wd_dtw_input(data)

mean_data = mean(data);
sigma = 0.01 * mean_data;
noise = sigma * randn(size(data));
noisy_data = data + noise;
diff_data = diff(noisy_data);

end