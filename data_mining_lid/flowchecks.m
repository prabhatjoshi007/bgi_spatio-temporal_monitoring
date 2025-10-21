%--------------------------------------------------------------------------
% Function: flowchecks
%
% Description:
%   Evaluates whether two rainfall-runoff events are suitable for 
%   hydrological comparison based on independence, interannual separation,
%   and soil moisture similarity.
%
%   Specifically, the function checks:
%     1. That no flow exists before rainfall onset (independence).
%     2. That the two events are from different years.
%     3. That the antecedent soil moisture (SM) values are within a 
%        defined similarity threshold.
%
% Syntax:
%   flowconditionsMet = flowchecks(rain1, rain2, flow1, flow2, year1, year2, SM1, SM2, SM_threshold)
%
% Inputs:
%   rain1, rain2     - Vectors of rainfall for event 1 (early period) and event 2
%   (later period)
%  
%   flow1, flow2     - Vectors of flow/discharge corresponding to rain1
%   (early) and rain2 (later period)
%   year1, year2     - Scalars indicating the year of each event
%   SM1, SM2         - Scalars representing antecedent soil moisture (e.g., averaged before event)
%   SM_threshold     - Scalar threshold for acceptable SM difference (e.g., 0.05)
%
% Output:
%   flowconditionsMet - Boolean (true/false). Returns true only if:
%                         - No flow occurs before rainfall starts in either event
%                         - Events are from different years
%                         - SM values are within the defined threshold
%
% Notes:
%   - The function returns false early with a message if any condition fails.
%   - This is a useful pre-check for hydrological similarity analyses or 
%     transfer learning applications between rainfall-runoff events.
%
% Example:
%   ok = flowchecks(rain_A, rain_B, flow_A, flow_B, 2010, 2012, 0.31, 0.33, 0.05);
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------

function flowconditionsMet = flowchecks(rain1, rain2, flow1, flow2, year1, year2, SM1, SM2, SM_threshold)
% Initialize default output values

idx1 = find(rain1, 1, 'first');
idx2 = find(rain2, 1, 'first');

if or(sum(flow1(1:idx1-1)) > 0, sum(flow2(1:idx2-1)) > 0)
    flowconditionsMet = false;
    fprintf('The event is not independent.\n')
    return
end

% Check if the flows are from the same year
if year1 == year2
    flowconditionsMet = false;
    fprintf('Same year condition met.\n')
    return
end

% Check if soil moisture values are similar within a given threshold
if abs(SM1 - SM2) > SM_threshold
    flowconditionsMet = false;
    fprintf('SM threshold not met.\n')
    return
end

flowconditionsMet = true;
% Additional statistical similarity checks can be added here if needed
end
