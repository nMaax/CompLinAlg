function first_eig = DeflationMethod(A, M)
    first_eig = zeros(M);
    for i = 1:M
        [eig_val, eig_vec] = InversePowerMethod(A, 0, 10, 0.0001);
        first_eig(i) = eig_val;
        A = DeflationProcess(A,eig_vec);
    end
end