clear all
% Parameters
N = 1000; % Number of grid points
h = 1 / (N - 1); % Step size
x = linspace(0, 1, N); % Grid points including boundaries
x=x';
% exact solution
u_exact=sin(pi*x);
% Tolerance and maximum iterations for (Quasi-)Newton's method
tol = 1e-10;
max_iter = 100;

u0x = x;

%The number of trial
num_iterations=1;
%Perturbation scale
scales = [0.1];
%Define nonlinear function and it's derivative
nonli = @(x) x.^3;
nonli_df= @(x) 3.*x.^2 ;


for scale=scales

for op=1 %If op=1 it is Newton if op=2 it is Quasi-Newton method
for repeat=1:num_iterations
%Define perturbation
perturbation=0;
for itt1=1:5
    u1x=u0x.^(itt1-1);
    perturbation=perturbation+(rand(1)*2-1)*u1x;
end
perturbation=perturbation/(max(abs(perturbation(:))));
perturbation=scale*perturbation;

u=u_exact+perturbation;


tic;

% (Quasi-)Newton's method loop
for iter = 1:max_iter
    % Compute F(u) and the Jacobian J
    F = zeros(N, 1);
    J = zeros(N, N);
    %Boundary condition
    u(1)=0;
    u(N)=0;
    for i = 1:N
        x_i = x(i);
        
        % Right-hand side
        rhs = h * (pi^2 * sin(pi * x_i) + sin(pi * x_i)^3);
        
        % Nonlinear function F_i(u)
        if i == 1 %Boundary condition
            F(i) =u(i)/h;
            J(i, i) = 1/h;
        elseif i == N  %Boundary condition
            F(i) =u(i)/h;
            J(i, i) = 1/h;
        else
            F(i) = (-u(i-1) + 2*u(i) - u(i+1))/h + h * nonli(u(i)) - rhs;
            J(i, i-1) = -1/h;
            J(i, i) = 2/h +  h * nonli_df(u(i));
            J(i, i+1) = -1/h;
        end


    end
    if op>1
        mx=max(nonli_df(u(:)));
        mi=min(nonli_df(u(:)));
    end
    if op==2
        %\Beta size
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mx+mi)/2));
        %J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mx+mi)));
        %J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mx+mx)/2));
        %J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*1);
    end


    % Solve for the (Quasi-)Newton step
    delta_u = -J \ F;
    
    % Update u
    u = u + delta_u;
    
    % Check for convergence
    if norm(F, inf) < tol
        break;
    end
end
% Check if the method did not converge
elapsed_time=toc;

if iter == max_iter
    warning('(Quasi-)Newton''s method did not converge.');
else
    fprintf('Converged after %d iterations when op is %g, perturbation as %g with time %g. The true error is %g.\n', iter,op,scale,elapsed_time,max(abs(u-u_exact)));
end

end

end

end

plot(x,u_exact)
hold on
plot(x,u,"--")
