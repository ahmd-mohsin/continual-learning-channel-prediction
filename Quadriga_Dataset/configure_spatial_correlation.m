function builder = configure_spatial_correlation(builder, SC_lambda, lsp_xcorr)
% CONFIGURE_SPATIAL_CORRELATION sets the spatial correlation parameters.
%
% Inputs:
%   builder   - QuaDRiGa channel builder object
%   SC_lambda - Decorrelation distance for small-scale fading (in meters). 
%               A value of 0 disables spatial consistency for SSF.
%   lsp_xcorr - (Optional) Correlation matrix for large-scale parameters.
%               If not provided, defaults to the identity matrix.
%
% Output:
%   builder   - Modified builder with updated spatial correlation parameters.
%
% Note: SC_lambda controls how quickly the SSF decorrelates with distance.

builder.scenpar.SC_lambda = SC_lambda;
if nargin < 3
    % Default: disable inter-parameter correlation by setting to identity.
    builder.lsp_xcorr = eye(8);
else
    builder.lsp_xcorr = lsp_xcorr;
end

end
