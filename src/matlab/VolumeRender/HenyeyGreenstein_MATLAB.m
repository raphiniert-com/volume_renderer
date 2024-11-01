function hg_lut = HenyeyGreenstein_LUT(N, g)
    % Henyey-Greenstein phase function LUT generator in MATLAB
    % N: LUT resolution (e.g., 64)
    % g: asymmetry factor (must be in the range [-1, 1])

    if nargin < 2
        g = 0.8;  % Default asymmetry factor
    end

    % Validate asymmetry factor g
    if g < -1 || g > 1
        error('Asymmetry factor g must be in the range [-1, 1]');
    end

    % Precompute g^2 for efficiency
    gSquared = g * g;

    % Allocate 3D LUT (N x N x N)
    hg_lut = zeros(N, N, N);

    % Angular step sizes
    frac_full = 2 * pi / N;  % For gamma angles
    frac_half = pi / N;      % For alpha and beta angles

    % Iterate over all dimensions to fill the LUT
    for c = 1:N
        % Gamma angle (rotation around X-axis)
        gamma = (c - 1) * frac_full;

        for a = 1:N
            % Alpha angle (lightOut angle)
            alpha = (a - 1) * frac_half;
            lightOut = [sin(alpha), 0, cos(alpha)];

            for b = 1:N
                % Beta angle (lightIn angle)
                beta = (b - 1) * frac_half;
                lightIn = [sin(beta), 0, cos(beta)];

                % Apply rotation around the X-axis by gamma
                R = [1, 0, 0; 0, cos(gamma), -sin(gamma); 0, sin(gamma), cos(gamma)];
                lightOut_rotated = (R * lightOut')';

                % Compute the dot product (cosine of the angle)
                cosTheta = dot(lightIn, lightOut_rotated);
                cosTheta = max(-1, min(1, cosTheta));  % Clamp to [-1, 1]

                % Henyey-Greenstein phase function calculation
                numerator = 1 - gSquared;
                denominator = (1 + gSquared - 2 * g * cosTheta)^(3/2);
                hg_value = (1 / (4 * pi)) * (numerator / denominator);

                % Store value in the LUT (column-major indexing to match MATLAB)
                hg_lut(a, b, c) = single(hg_value);
            end
        end
    end
end
