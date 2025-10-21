%--------------------------------------------------------------------------
% Function: tip_to_uniform_flow
%
% Description:
%   Converts a discrete tipping bucket time series into an estimated 
%   uniform flow rate signal. This is commonly used to transform 
%   precipitation or flow data from a tipping bucket rain gauge into a 
%   time-continuous representation of uniform intensity over the interval 
%   between tips.
%
%   The function:
%     - Finds indices of non-zero tips (e.g., 0.2 mm per tip)
%     - Estimates uniform flow rate between tips as:
%         flow = tip volume / time interval
%     - Fills values forward to simulate a uniform contribution
%     - Sets post-tip trailing values to zero
%
% Syntax:
%   UniFlow = tip_to_uniform_flow(data)
%
% Input:
%   data - A numeric column vector representing tip volume per timestep.
%          Typically consists of mostly zeros with isolated non-zero values 
%          indicating a tipping event.
%
% Output:
%   UniFlow - A column vector of the same size as `data`, representing the 
%             estimated uniform flow rate between each pair of tips.
%
% Notes:
%   - If two consecutive tips are equal in value, the value at the first tip 
%     is preserved directly in one step (legacy line; may be redundant).
%   - The last segment (after the last tip) is set to 0.
%   - `fillmissing(..., 'previous')` is used to propagate flow values across 
%     the uniform intervals.
%
% Example:
%   % Simulate tips at t=2, 5, 9 with 0.2 mm/tip
%   data = [0; 0.2; 0; 0; 0.2; 0; 0; 0; 0.2; 0];
%   flow = tip_to_uniform_flow(data);
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------


function [UniFlow] = tip_to_uniform_flow(data)

UniFlow = nan(size(data,1),1);

% Find the indices of non-zero values
UniFlow = nan(size(data,1),1);
nonZeroIndices = find(data ~= 0);

% Iterate through non-zero values
for i = 1:numel(nonZeroIndices)-1
    startIndex = nonZeroIndices(i);
    endIndex = nonZeroIndices(i+1);
    if data(startIndex) == data(endIndex)
        UniFlow(startIndex) = data(endIndex);
    end
  
end

for j = 1:numel(nonZeroIndices)-1
    startIndex = nonZeroIndices(j);
    endIndex = nonZeroIndices(j+1);

    UniFlow(startIndex) = abs(data(startIndex))/(endIndex - startIndex);
    
end

startRow = nonZeroIndices(end);
endRow = numel(data);

% Identify the NaN values within the specified range
nanIndices = isnan(UniFlow(startRow:endRow));

% Replace NaN values with zeros within the specified range
UniFlow(startRow + find(nanIndices) - 1) = 0;

UniFlow = fillmissing(UniFlow,"previous");
UniFlow = fillmissing(UniFlow,'constant',0);

end