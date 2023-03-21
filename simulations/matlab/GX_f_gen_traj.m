function [x, y, z, m_adAzimuthalAngle, m_adPolarAngle] = GX_f_gen_traj(m_lProjectionsPerFrame, m_lNumberOfFrames, option)
% Input
% option: 1. spiral; 2. halton sequence; 3. haltonized spiral; 4.
% archimedean spiral; 5. double golden mean
%
%Define output
lTotalNumberOfProjections = m_lProjectionsPerFrame * m_lNumberOfFrames;
m_adPolarAngle = zeros(1, lTotalNumberOfProjections); %array for polar angle
m_adAzimuthalAngle = zeros(1, lTotalNumberOfProjections); %array for azimuthal angle

%%
switch option
    case 1 %Spiral Method
        dPreviousAngle = 0;
        for lk = 0:1:m_lProjectionsPerFrame - 1 % index for projection
            for lFrame = 0:1:m_lNumberOfFrames - 1 % number of frames per projection
                llin = lFrame + lk * m_lNumberOfFrames;
                linter = lk + lFrame * m_lProjectionsPerFrame + 1;
                dH = -1 + 2 * llin / lTotalNumberOfProjections;

                %%
                m_adPolarAngle(linter) = acos(dH);
                if (llin == 0)
                    m_adAzimuthalAngle(linter) = 0;
                else
                    m_adAzimuthalAngle(linter) = mod(dPreviousAngle+3.6/(sqrt(lTotalNumberOfProjections*(1 - dH * dH))), 2.0*pi);
                end
                dPreviousAngle = m_adAzimuthalAngle(linter);
            end
        end
    case 2 % Halton Sequence
        p1 = 2;
        p2 = 3;
        for lk = 0:1:m_lProjectionsPerFrame - 1 % index for projection
            for lFrame = 0:1:m_lNumberOfFrames - 1 % number of frames per projection
                linter = lk + lFrame * m_lProjectionsPerFrame + 1;

                z = haltonnumber(lk+1, p1) * 2 - 1;
                phi = 2 * pi * haltonnumber(lk+1, p2);

                %% changed to -z
                m_adPolarAngle(linter) = acos(z);
                m_adAzimuthalAngle(linter) = phi;
            end
        end
    case 3 % Haltonized Spiral
        if (m_lNumberOfFrames > 1)
            warning('number of frame > 1 for haltonized spiral, unhappy');
        end
        [~, ~, ~, ht_azi, ht_polar] = GX_f_gen_traj(m_lProjectionsPerFrame, 1, 2);
        [~, ~, ~, sp_azi, sp_polar] = GX_f_gen_traj(m_lProjectionsPerFrame, 1, 1);
        [m_adAzimuthalAngle, m_adPolarAngle] = GX_f_haltonedSpiral(ht_polar, sp_azi, sp_polar, m_lProjectionsPerFrame);
    case 4 % Archimedean Spiral
        primeplus = 180 * (3 - sqrt(5));
        dangle = primeplus * (pi / 180);
        dz = 2 / (m_lProjectionsPerFrame - 1);

        for lk = 0:1:m_lProjectionsPerFrame - 1 % index for projection
            for lFrame = 0:1:m_lNumberOfFrames - 1 % number of frames per projection
                linter = lk + lFrame * m_lProjectionsPerFrame + 1;
                z = 1 - dz * lk;
                m_adPolarAngle(linter) = acos(z);
                m_adAzimuthalAngle(linter) = lk * dangle;
            end
        end
    case 5 % double golden mean
        goldmean1 = 0.465571231876768;
        goldmean2 = 0.682327803828019;

        for lFrame = 0:1:m_lNumberOfFrames - 1
            for lk = 0:1:m_lProjectionsPerFrame
                linter = lk + lFrame * m_lProjectionsPerFrame + 1;
                m_adPolarAngle(linter) = acos(2.0*mod(lk*goldmean1, 1)-1);
                m_adAzimuthalAngle(linter) = 2 * pi * mod(lk*goldmean2, 1);
            end
        end
    case 6 % tiny golden mean
        goldmean1 = 0.465571231876768;
        goldmean2 = 0.682327803828019;
        N = 15;
        phi_N_2 = 1/(N -1 +1/goldmean2);
        phi_N_1 = phi_N_2/(1+goldmean1);
        for lFrame = 0:1:m_lNumberOfFrames - 1
            for lk = 0:1:m_lProjectionsPerFrame
                linter = lk + lFrame * m_lProjectionsPerFrame + 1;
                m_adPolarAngle(linter) = acos(2.0*mod(lk*phi_N_1, 1)-1);
                m_adAzimuthalAngle(linter) = 2 * pi * mod(lk*phi_N_2, 1);
            end
        end
    otherwise
        warning('Unexpected trajectory type key.');
end

%% visualization
n_ray = m_lProjectionsPerFrame;
m_polar = m_adPolarAngle(1:1:n_ray);
m_azi = m_adAzimuthalAngle(1:1:n_ray);

%% convert to 3D x-y-z
x = zeros(size(m_polar));
y = zeros(size(m_polar));
z = zeros(size(m_polar));
for j = 1:1:n_ray
    x(j) = sin(m_polar(j)) * cos(m_azi(j));
    y(j) = sin(m_polar(j)) * sin(m_azi(j));
    z(j) = cos(m_polar(j));
end
