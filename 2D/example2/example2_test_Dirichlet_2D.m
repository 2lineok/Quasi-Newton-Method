clear all
close all
addpath("Functions")
%This part is setting the parameter main code is after this setting.
for parameter_setting=1
    % Solving the Poisson equation by Q5 SEM with Neumann b.c.
    if gpuDeviceCount('available')<1; Param.device='cpu';
    else;Param.device='gpu'; Param.deviceID=1;end % ID=1,2,3,...
    Np=5; Param.Np=Np; % polynomial degree Q5
    Ncell=10; % finite element cell number
    nx=Ncell*Np+1; ny=Ncell*Np+1;
    BC='Dirichlet';% BC='NeumannBC';% BC='Periodic_';
    Left=0;
    Right=1;
    [n, x,dx, H, M, S,T,invT,lambda]= SEGenerator1D_2(Left,Right,Ncell, Np, BC);
    fprintf('2D Poisson with total DoFs %d by %d \n',n,n);
    fprintf('Laplacian is Q%d spectral element method \n', Np);  
    tol=10^(-9);
    [X, Y] = meshgrid(x, x);
    cx=2;
    cy=1;
    uexact=squeeze((sin(cx*pi*x)*sin(cy*pi*x)'));
    Z=uexact;
    f=((cx^2+cy^2)*pi^2+2)*uexact+uexact.^2;
    ex=ones(n,1);

    if strcmp(Param.device,'gpu');Device=gpuDevice(Param.deviceID);
       fprintf('GPU computation: starting to load matrices/data \n');
       Tx=gpuArray(T); Ty=gpuArray(T);lambda=gpuArray(lambda);
       ex=gpuArray(ex); ey=gpuArray(ex); f=gpuArray(f);  x=gpuArray(x); 
       invTx=gpuArray(invT); invTy=gpuArray(invT); Z=gpuArray(Z);
    end 
    %This is our Lapalacian operator's eigenvalues
    Lambda2D=squeeze(tensorprod(lambda,ey)+tensorprod(ex,lambda));
    linear_matrix=2*eye((ny-2)*(nx-2))+kron(eye(ny-2),Tx*diag(lambda)*invTx)+kron(Ty*diag(lambda)*invTy,eye(nx-2));

end



for scale = 1 %Perturbation scale

u0x = x;
u0y = x;
%Define perturbation
perturbation=0;
for itt1=1:5
for itt2=1:5
u1x=u0x.^(itt1-1);
u1y=u0y.^(itt2-1);
perturbation=perturbation+(rand(1)*2-1)*squeeze(u1x*u1y');
end
end
perturbation=perturbation/(max(abs(perturbation(:))));
perturbation=scale*perturbation;
u0 = uexact + perturbation;


for op = 1
    figure
    subplot(1,2,1)
    
    % Create a surface plot
    surf(X, Y, Z, 'EdgeColor', 'none');
    title("Exact solution")
    grid off;
    colorbar;
    tic;
    fprintf("this is op is %d \n", op);

    u0 = Z + perturbation;
    u_vector=u0(:);
    for i = 1:500
        if op==1
            % Computing the residual part F
            jmatrix=linear_matrix+diag(2.*u_vector);
            res=linear_matrix*u_vector+u_vector.^2;
            res = res - f(:);      
            u_vector=u_vector-jmatrix\res(:);
        else
        % Computing the residual part F
        u1 = tensorprod(u0, invTy', 2, 1);
        u1 = squeeze(tensorprod(invTx, u1, 2, 1));
        u1 = u1 .* (Lambda2D+2);
        u1 = tensorprod(u1, Ty', 2, 1);
        u1 = squeeze(tensorprod(Tx, u1, 2, 1));
        u1 = u1 + u0.^2;
        res = u1 - f;
        %Update the beta
        beta = 2+(max(max(max(2 * u0))) + min(min(min(2 * u0)))) / 2;
        %Quasi-Newton %Computing J\F
        u = tensorprod(res,invTy',2,1);
        u = squeeze(tensorprod(invTx,u,2,1));
        u = u./(Lambda2D + beta);
        u = tensorprod(u,Ty',2,1);
        u = squeeze(tensorprod(Tx,u,2,1));
        %Update u
        u0=u0-u;
        end
        if norm(res(:), inf) < tol
            if op==1
                u0=u_vector.reshape(size(u0));
            end
            break
        end
    end
    elapsed_time = toc;
    fprintf("After %d iteration \n", i);
    fprintf("residual is %d \n", norm(res(:), inf));
    fprintf("error is %d \n", norm(u0(:)-uexact(:), inf));
subplot(1,2,2)
% Create a surface plot
surf(X, Y, u0, 'EdgeColor', 'none');
grid off;
title("Numerical solution")
colorbar;
end

end
