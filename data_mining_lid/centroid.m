%%
%--------------------------------------------------------------------------
% Function: centroid
%
% Description:
%   Computes the temporal centroid (center of mass) of a time series vector.
%   The centroid is calculated as the weighted average time index, where
%   weights are the magnitudes of the signal at each time step.
%
% Syntax:
%   centroid_time = centroid(x)
%
% Input:
%   x - A numeric column or row vector representing the signal or time series.
%       Typically corresponds to a hydrograph, pollutograph, or similar data.
%
% Output:
%   centroid_time - A scalar representing the centroid time (index) of the input series.
%
% Notes:
%   - Time is assumed to increase linearly with a step size of 1.
%   - This function treats the input signal as a discrete mass distribution.
%
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Last updated on: 2025-04-30
%--------------------------------------------------------------------------

function centroid_time = centroid(x)

t = 1:1:length(x);
centroid_time = sum(x.*t')/sum(x);

end
