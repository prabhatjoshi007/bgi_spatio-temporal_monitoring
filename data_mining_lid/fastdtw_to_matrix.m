%--------------------------------------------------------------------------
% Function: fastdtw_to_matrix
%
% Description:
%   Converts the warping path output from a Python FastDTW function
%   (typically a cell array containing a path of tuples) into a
%   MATLAB-compatible numeric matrix of coordinates.
%
%   This is useful when interfacing MATLAB with Python DTW libraries,
%   enabling further analysis or plotting of alignment paths.
%
% Syntax:
%   pathMatrix = fastdtw_to_matrix(x)
%
% Input:
%   x - A 1x2 cell array where x{2} is a cell array of 2-element tuples
%       representing the DTW path. Each tuple contains two indices
%       corresponding to aligned positions in the two time series.
%
% Output:
%   pathMatrix - An N×2 numeric matrix where each row contains the
%                (i, j) index pair from the DTW path.
%
% Notes:
%   - Python uses 0-based indexing; this function assumes such input.
%     If indices need to be converted to MATLAB 1-based indexing,
%     you can simply add 1 to pathMatrix after this function.
%
% Example:
%   % Given a Python FastDTW output `py_out`:
%   pathMatrix = fastdtw_to_matrix(py_out);
%
% Author: Prabhat Joshi; prabhat.joshi@eawag.ch
% Late updated on: 2025-04-30
%--------------------------------------------------------------------------


function pathMatrix = fastdtw_to_matrix(x)
path = x{2};  % Python indices start from 0, MATLAB from 1 adjusted to 0

% Initialize an empty matrix to store coordinates
pathMatrix = zeros(length(path), 2);  % Pre-allocate for efficiency

% Convert each Python tuple to MATLAB row vector
for i = 1:length(path)
    currentTuple = path{i};  % Adjust index for Python (0-based)
    pathMatrix(i, 1) = double(currentTuple{1});  % Extract x-coordinate
    pathMatrix(i, 2) = double(currentTuple{2});  % Extract y-coordinate
end

end