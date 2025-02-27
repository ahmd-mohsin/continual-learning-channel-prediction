function builder = configure_angles(builder, perClusterAS_A, perClusterAS_D, perClusterES_A, perClusterES_D)
% CONFIGURE_ANGLES sets the per-cluster angular spread parameters.
%
% Inputs:
%   builder         - QuaDRiGa channel builder object
%   perClusterAS_A  - Per-cluster azimuth spread at the transmitter (degrees)
%   perClusterAS_D  - Per-cluster azimuth spread at the receiver (degrees)
%   perClusterES_A  - Per-cluster elevation spread at the transmitter (degrees)
%   perClusterES_D  - Per-cluster elevation spread at the receiver (degrees)
%
% Output:
%   builder         - Modified builder with updated angular spread parameters.
%
% Note: These parameters limit the angular spread of each scattering cluster.

builder.scenpar.PerClusterAS_A = perClusterAS_A;
builder.scenpar.PerClusterAS_D = perClusterAS_D;
builder.scenpar.PerClusterES_A = perClusterES_A;
builder.scenpar.PerClusterES_D = perClusterES_D;

end
