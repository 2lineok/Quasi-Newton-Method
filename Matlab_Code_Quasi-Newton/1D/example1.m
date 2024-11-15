clear all
results = struct('Method', {}, 'Iterations', {}, 'Time', {});
% Parameters
N = 1000; % Number of interior points
h = 1 / (N - 1); % Step size
x = linspace(0, 1, N); % Grid points including boundaries
x=x';
% Initial guess for u (starting with zero or any other initial guess)
u = zeros(N, 1);
u_exact=sin(pi*x);
% Tolerance and maximum iterations for Newton's method
tol = 1e-10;
max_iter = 100;

u0x = x;


num_iterations=100;
scales = [0.1,0.2,0.5,1.0];
%defome nonlinear function
nonli = @(x) x.^3;
nonli_df= @(x) 3.*x.^2 ;


for scale=scales

for op=1:10
    iterations = zeros(num_iterations, 1);
    times = zeros(num_iterations, 1);
for repeat=1:num_iterations


perturbation=0;
for itt1=1:5
    u1x=u0x.^(itt1-1);
    perturbation=perturbation+(rand(1)*2-1)*u1x;
end
perturbation=perturbation/(max(abs(perturbation(:))));
perturbation=scale*perturbation;

u=u_exact+perturbation;


tic;

% Newton's method loop
for iter = 1:max_iter
    % Compute F(u) and the Jacobian J
    F = zeros(N, 1);
    J = zeros(N, N);
    u(1)=0;
    u(N)=0;
    for i = 1:N
        % x_i for the current node
        x_i = x(i); % +1 because x includes boundaries
        
        % Right-hand side
        rhs = h * (pi^2 * sin(pi * x_i) + sin(pi * x_i)^3);
        
        % Nonlinear function F_i(u)
        if i == 1
            F(i) =u(i)/h;
            J(i, i) = 1/h;
        elseif i == N
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
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mx+mi)/2));
    elseif op==3
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mx+mi)));
    elseif op==4
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mx+mx)/2));
    elseif op==5
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*((mi+mi)/2));
    elseif op==6
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*0.1);
    elseif op==7
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*0.5);
    elseif op==8
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*1);
    elseif op==9
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*5);
    elseif op==10
        J(2:end-1,2:end-1)=J(2:end-1,2:end-1)-diag((h).*nonli_df(u(2:end-1)))+eye(N-2).*(h*10);
    end


    % Solve for the Newton step
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
iterations(repeat) = iter;
times(repeat) = elapsed_time;

if iter == max_iter
    warning('Newton''s method did not converge.');
    op
    iter
else
fprintf('Converged after %d iterations when op is %g and perturbation as %g with time %g error is %g.\n', iter,op,scale,elapsed_time,max(abs(u-u_exact)));
end


end
avg_iterations = mean(iterations);
std_iterations = std(iterations);

avg_times = mean(times);
std_times = std(times);

results(op).Method = sprintf('Method %d', op);
results(op).Iterations = [results(op).Iterations, {strcat(sprintf('%.1f ',avg_iterations),' $\\pm$ ',sprintf(' %.1f',std_iterations))}];
results(op).Time = [results(op).Time, {strcat(sprintf('%.2f ',avg_times),' $\\pm$ ',sprintf(' %.2f',std_times))}];

end


end



latex_methods = { ...
    'Newtons method', ...
    '$\\beta=\\frac{min+max}{2}$', ...
    '$\\beta={min+max}$', ...
    '$\\beta=max$', ...
    '$\\beta=min$', ...
    '$\\beta=0.1$', ...
    '$\\beta=0.5$', ...
    '$\\beta=1$', ...
    '$\\beta=5$', ...
    '$\\beta=10$', ...
};


latex_table = '\\begin{table}[h!] \n \\centering \n \\begin{tabular}{|c|c|c|c|c|c|} \n \\hline \n';
latex_table = [latex_table, '\\multirow{2}{*}{Method} & \\multirow{2}{*}{Metric} & \\multicolumn{4}{c|}{Perturbation Scale} \\\\ \\cline{3-6}\n'];
latex_table = [latex_table, ' & & 0.1 & 0.2 & 0.5 & 1 \\\\ \\hline'];

for i = 1:length(results)
    method =latex_methods{i};
    iterations = results(i).Iterations;
    time = results(i).Time;
    
    % Add method row with multirow for iterations and time
    latex_table = [latex_table, '\\multirow{2}{*}',sprintf('{%s} & Iterations & ', method),strjoin(iterations, ' & ')];
    latex_table = [latex_table,'\\\\ \\cline{2-6} \n'];
    latex_table = [latex_table, ' & Time (s) & ', strjoin(results(i).Time, ' & ')];
    latex_table = [latex_table,'\\\\ \\hline \n'];
end


latex_table = [latex_table, '\\end{tabular}\n\\caption{Convergence results for different methods}\n\\label{table:results1D}\n\\end{table}\n'];

% Write LaTeX table to file
fid = fopen('result1d.tex', 'w');
fprintf(fid, latex_table);
fclose(fid);
