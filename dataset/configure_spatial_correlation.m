function builder = configure_spatial_correlation(builder, SC_lambda, lsp_xcorr)
    % CONFIGURE_SPATIAL_CORRELATION sets the spatial correlation parameters.
    %
    % Inputs:
    %   builder   - QuaDRiGa channel builder object (scalar or array)
    %   SC_lambda - Decorrelation distance for small-scale fading (in meters).
    %               A value of 0 disables spatial consistency for SSF.
    %   lsp_xcorr - (Optional) Correlation matrix for large-scale parameters.
    %               If not provided, defaults to the identity matrix.
    %
    % Output:
    %   builder   - Modified builder with updated spatial correlation parameters.
    %
    % Note: SC_lambda controls how quickly the SSF decorrelates with distance.

    if nargin < 3
        lsp_xcorr = eye(8);
    end

    if isscalar(builder)
        builder.scenpar.SC_lambda = SC_lambda;
        builder.lsp_xcorr = lsp_xcorr;
    else

        for idx = 1:numel(builder)
            builder(idx).scenpar.SC_lambda = SC_lambda;
            builder(idx).lsp_xcorr = lsp_xcorr;
        end

    end

end
