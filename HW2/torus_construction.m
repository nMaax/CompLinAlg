R1 = 3; 
r1 = 1; 

R2 = 7; 
r2 = 1; 

theta1 = 2 * pi * rand(300, 1);
phi1 = 2 * pi * rand(300, 1);
theta2 = 2 * pi * rand(600, 1);
phi2 = 2 * pi * rand(600, 1);

X1 = (R1 + r1 * cos(phi1)) .* cos(theta1);
Y1 = (R1 + r1 * cos(phi1)) .* sin(theta1);
Z1 = r1 * sin(phi1);

X2 = (R2 + r2 * cos(phi2)) .* cos(theta2);
Y2 = (R2 + r2 * cos(phi2)) .* sin(theta2);
Z2 = r2 * sin(phi2);

% Combinazione dei dati
X = [X1; X2];
Y = [Y1; Y2];
Z = [Z1; Z2];
% Combinare le coordinate in una matrice
data = [X, Y, Z];
