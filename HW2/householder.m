function H = householder(x)
    v = x(:);  

    % Calcolo di sigma
    sigma = sign(v(1)) * norm(v);

    % Calcolo di u (versore)
    u = v + sigma * eye(length(v), 1);
    u = u / norm(u);

    % Calcolo della matrice di riflessione
    H = eye(length(u)) - 2 * (u * u');

    return;
end