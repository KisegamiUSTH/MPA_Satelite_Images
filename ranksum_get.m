% MPA best fitness values (excluding values < -1)
MPA_fitness = [-0.083854, -0.11339, -0.060786, -0.070671, -0.097564, -0.081894, -0.045749, ...
    -0.099915, -0.10678, -0.059898, -0.088024, -0.095652, -0.076781, -0.073872, ...
    -0.06934, -0.049856, -0.062272, -0.10194, -0.056624, -0.054741];

% WOA best fitness values (excluding values < -1)
WOA_fitness = [-0.041631, -0.018287, -0.078962, -0.029387, -0.063942, -0.070852, -0.041587, ...
    -0.050534, -0.047371, -0.064809, -0.064214, -0.092183, -0.092183, -0.085163, ...
    -0.046043, -0.043193, -0.021506, -0.035336, -0.046474, -0.033224];

% SCA best fitness values (excluding extreme value < -1)
SCA_fitness = [0.24477, -0.0014034, 0.13844, -0.015702, -0.0050263, -0.01009, 0.20205, ...
    -0.001313, -0.025475, 0.20013, -0.0055053, 0.00019861, 0.0087716, -0.023858, ...
    -0.027737, 0.050887, -0.012689, -0.00022074, -0.026956];

% SSA best fitness values (excluding values < -1)
SSA_fitness = [0.15987, 0.51961, 0.30058, 0.098142, 0.52746, 0.35078, 0.03325, 0.26627, ...
    0.29563, 0.134, 0.13248, 0.26437, 0.0015098, 0.38386, 0.41215, 0.13947, ...
    0.45621, 0.013412];

% GA best fitness values (excluding values < -1)
GA_fitness = [-0.0095558, -0.013546, -0.0087601, -0.034088, -0.011668, -0.041086, -0.0078429, ...
    -0.028308, -0.010291, -0.03972, -0.0089708, -0.016387, -0.028583, -0.011539, ...
    -0.012665, -0.016889, -0.025816, -0.01242, -0.0049565, -0.008359];

% MVO best fitness values (excluding values < -1)
MVO_fitness = [-0.16401, -0.16262, -0.16309, -0.16398, -0.15686, -0.16323, -0.16097, ...
    -0.16222, -0.16058, -0.16222, -0.16251, -0.15487, -0.1648, -0.16258, ...
    -0.16455, -0.16134, -0.15761];

% DA best fitness values (excluding values < -1)
DA_fitness = [-0.0036674, -0.0088169, -0.02933, -0.027217, -0.025154, -0.007968, -0.011528, ...
    -0.0081108, -0.015292, -0.0035019, -0.02549, -0.014748, -0.0099225, -0.00757, ...
    -0.0081228, -0.024415, -0.056482, -0.021606];

% Perform Wilcoxon rank-sum test for each comparison
[p_WOA, h_WOA] = ranksum(MPA_fitness, WOA_fitness);
disp(['P-value for MPA vs WOA: ', num2str(p_WOA)]);

[p_SCA, h_SCA] = ranksum(MPA_fitness, SCA_fitness);
disp(['P-value for MPA vs SCA: ', num2str(p_SCA)]);

[p_SSA, h_SSA] = ranksum(MPA_fitness, SSA_fitness);
disp(['P-value for MPA vs SSA: ', num2str(p_SSA)]);

[p_GA, h_GA] = ranksum(MPA_fitness, GA_fitness);
disp(['P-value for MPA vs GA: ', num2str(p_GA)]);

[p_MVO, h_MVO] = ranksum(MPA_fitness, MVO_fitness);
disp(['P-value for MPA vs MVO: ', num2str(p_MVO)]);

% Add the rank sum test for DA
[p_DA, h_DA] = ranksum(MPA_fitness, DA_fitness);
disp(['P-value for MPA vs DA: ', num2str(p_DA)]);
