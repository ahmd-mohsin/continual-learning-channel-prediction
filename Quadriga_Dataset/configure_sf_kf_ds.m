function builder = configure_sf_kf_ds(builder, KF_mu, KF_sigma, DS_us, DS_sigma, SF_sigma)
% CONFIGURE_SF_KF_DS sets the shadow fading, K-Factor, and RMS delay spread parameters.
%
% Inputs:
%   builder   - QuaDRiGa channel builder object
%   KF_mu     - Mean K-Factor (in dB)
%   KF_sigma  - Standard deviation of the K-Factor (in dB)
%   DS_us     - Median RMS delay spread in microseconds (will be converted to log10(seconds))
%   DS_sigma  - Standard deviation of log10(delay spread)
%   SF_sigma  - Standard deviation of shadow fading (in dB)
%
% Output:
%   builder   - Modified builder with updated SF, KF, and DS parameters.
%
% Note: DS_us is converted from microseconds to seconds then to log10 scale.

builder.scenpar.KF_mu = KF_mu;
builder.scenpar.KF_sigma = KF_sigma;
builder.scenpar.DS_mu = log10(DS_us * 1e-6); % convert microseconds to seconds and take log10
builder.scenpar.DS_sigma = DS_sigma;
builder.scenpar.SF_sigma = SF_sigma;

end
