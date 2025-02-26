clear all
close all
addpath("Functions")
%This part is setting the parameter main code is after this setting.
for parameter_setting=1
    % Solving the Poisson equation by Q5 SEM with Neumann b.c.
    if gpuDeviceCount('available')<1; Param.device='cpu';
    else;Param.device='gpu'; Param.deviceID=1;end % ID=1,2,3,...
    Np=5; Param.Np=Np; % polynomial degree Q5
    Ncellx=10; Ncelly=10; % finite element cell number
    % total number of unknowns in each direction
    nx=Ncellx*Np+1; ny=Ncelly*Np+1;
    % the domain is [-Lx, Lx]*[-Ly, Ly]
    Lx=1; Ly=1; Lz=1; cx=pi; cy=2*pi;alpha=1;
    Param.Ncellx=Ncellx; Param.Ncelly=Ncelly;
    Param.nx = nx; Param.ny = ny;
    fprintf('2D example 1 with total DoFs %d by %d \n',nx,ny);
    %x is location ex is 1 or identity and Tx is T and eigx is lambdax in paper
    [x,ex,Tx,eigx]=SEGenerator1D('x',Lx,Param);
    [y,ey,Ty,eigy]=SEGenerator1D('y',Ly,Param);
    u1x=cos(cx*x);
    u1y=cos(cy*y);
    tol=10^(-9);
    %exact solution
    uexact=squeeze((u1x*u1y'));
    %Right hand side
    f=(cx*cx+cy*cy)*squeeze((u1x*u1y'))+alpha*uexact+uexact.^3;
    TxInv=pinv(Tx); TyInv=pinv(Ty);
   Tx_matrix = Tx;
   Ty_matrix = Ty;
   TxInv_matrix = TxInv;
   TyInv_matrix = TyInv;
   linear_matrix=alpha*eye(ny*nx)+kron(eye(ny),Tx*diag(eigx)*TxInv)+kron(Ty*diag(eigy)*TyInv,eye(nx));
    if strcmp(Param.device,'gpu');Device=gpuDevice(Param.deviceID);
       fprintf('GPU computation: starting to load matrices/data \n');
       Tx=gpuArray(Tx); Ty=gpuArray(Ty);
       eigx=gpuArray(eigx);eigy=gpuArray(eigy);
       ex=gpuArray(ex); ey=gpuArray(ey); f=gpuArray(f); 
       TxInv=gpuArray(TxInv);TyInv=gpuArray(TyInv);
    end 
    %This is our Lapalacian operator's eigenvalues
    Lambda2D=squeeze(tensorprod(eigx,ey)+tensorprod(ex,eigy));
    if strcmp(Param.device,'gpu'); wait(Device);
        fprintf('GPU loading finished and computing started \n');
    end
    
    maxit=500; %Maximum iteration for iteration method
    
    scales = [0.1]; %Perturbation scale
end

for scale = scales

u0x = x;
u0y = y;

%The number of trial
num_iterations=1;

for op = 1 %If op=1 it is Newton if op=2 it is Quasi-Newton method
    iterations = zeros(num_iterations, 1);
    times = zeros(num_iterations, 1);

for iter=1:num_iterations
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
    tic;

    u_vector=u0(:);
    for i = 1:maxit
        
        if op==1
            % Computing the residual part F
            jmatrix=linear_matrix+diag(3*u_vector.^2);
            er=linear_matrix*u_vector+u_vector.^3;
            er = er - f(:);      
            u_vector=u_vector-jmatrix\er(:);
        elseif op==2
            % Computing the residual part F
            u1 = tensorprod(u0, TyInv', 2, 1);
            u1 = squeeze(tensorprod(TxInv, u1, 2, 1));
            u1 = u1 .* (Lambda2D + alpha);
            u1 = tensorprod(u1, Ty', 2, 1);
            u1 = squeeze(tensorprod(Tx, u1, 2, 1));
            u1 = u1 + u0.^3;
            er = u1 - f;
            %Update the beta
            beta = alpha + (max(max(max(3 * u0.^2))) + min(min(min(3 * u0.^2)))) / 2;
            %Quasi-Newton %Computing J\F
            u = tensorprod(er,TyInv',2,1);
            u = squeeze(tensorprod(TxInv,u,2,1));
            u = u./(Lambda2D + beta);
            u = tensorprod(u,Ty',2,1);
            u = squeeze(tensorprod(Tx,u,2,1));
            %Update u
            u0=u0-u;
        end
        if norm(er(:), inf) < tol || any(isnan(er(:)))
            if op==1
                u0 = reshape(u_vector, size(u0));
            end
            break
        end
        
        
    end
    elapsed_time = toc;
    fprintf("After %d iteration \n", i);
    fprintf("residual is %d \n", norm(er(:), inf));
    fprintf("true error is is %d \n", norm(u0(:)-uexact(:), inf));

end

end

end




[xx,yy]=meshgrid(x,y);

figure
subplot(1,2,1)
surf(xx, yy, uexact, 'EdgeColor', 'none');
xlabel('X');
ylabel('Y');
grid off;
title("Exact Solution")
colorbar;

subplot(1,2,2)
surf(xx, yy, u0, 'EdgeColor', 'none');
xlabel('X');
ylabel('Y');
grid off;
title("Numerical Solution")
colorbar;



