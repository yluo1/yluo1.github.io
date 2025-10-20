function [Q, R, P, r] = QR_mgs_p0(A, tol)
%Modified Gram Schmidt with column pivoting without data swaps
%--Yuancheng Luo, 2014

%A(:, P) = Q(:,P) * R(:,P), where A is MxN matrix 
%Q is orthonormal
%R(:,P) is upper triangular
%P is vector of permutations

%Output:
%Terminates when max-norm of remaining columns in subspace < tol

%For rank r, if r < N, then 
%A(:, P(1:r)) = Q(:,P(1:r)) * R(1:r, P(1:r))
%or A(:, P) approx Q(:, P(1:r)) * R(1:r, P)
if nargin < 2
    tol = 1e-6;
end

N = size(A, 2); 
Q = A;
R = zeros(N, N);
P = 1:N;
qnorms = sqrt(sum(Q.^2, 1));

for i = 1:N
    %Pivot selection
    [tmp, idx] = sort(qnorms(P(i:N)), 'descend');
    P(i:N) = P((i-1) + idx);    
    
    if(qnorms(P(i)) < tol)
        r = i-1;
        return;
    end
    
    %Q(:,P(i)) is fixed
    Q(:, P(i)) = Q(:, P(i)) / qnorms(P(i));
    
    e = Q(:, P(i));
	R(i,P(i:N)) = e' * A(:, P(i:N)); %Matrix-vector product
    for j = (i+1):N
        if qnorms(P(j)) >= tol %only update active-columns
            Q(:, P(j)) = Q(:, P(j)) - R(i,P(j)) * e; %Vector subtraction        
            qnorms(P(j)) = sqrt(Q(:, P(j))' * Q(:, P(j))); %Recompute norm
        end
    end
end
r = i;