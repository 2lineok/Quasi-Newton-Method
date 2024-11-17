clear all
%This file have a numerical solution when DOF is 50x50 and we will
%interpolate to get a finer solutions
load("numerical_solutions.mat")
%This part is setting the parameter main code is after this setting.
for parameter_setting=1
    sol_set=gather(sol_set);
    repeatnum=1;
    if gpuDeviceCount('available')<1; Param.device='cpu';
    else;Param.device='gpu'; Param.deviceID=1;end % ID=1,2,3,...

    %This is 50x50 mesh and we will try to interpolate to finer grid
    Np=5; Param.Np=Np; % polynomial degree Q5
    Ncellx=10; Ncelly=10; % finite element cell number
    % total number of unknowns in each direction
    nx=Ncellx*Np+1; ny=Ncelly*Np+1;
    % the domain is [-Lx, Lx]*[-Ly, Ly]*[-Lz, Lz]
    Lx=1/2; Ly=1/2; DA=2.5*10^(-4); DS=5*10^(-4); mu=0.065; rho=0.04;maxiteration=30000;tol=10^(-9); maxtol=1000;
    Param.Ncellx=Ncellx; Param.Ncelly=Ncelly;
    Param.nx = nx; Param.ny = ny;
    %x is location ex is 1 or identity and Tx is t and eigx is lambdax in paper
    %or we can understand as diagonal matrix 
    [x,ex,Tx,eigx]=SEGenerator1D('x',Lx,Param);
    [y,ey,Ty,eigy]=SEGenerator1D('y',Ly,Param);
    x=x+1/2;
    y=y+1/2;
    [X, Y] = meshgrid(x, y);
    
    
    
    %This is finer grid.
    Np=5; Param.Np=Np; % polynomial degree Q5
    Ncellx=20; Ncelly=20; % finite element cell number
    % total number of unknowns in each direction
    nx=Ncellx*Np+1; ny=Ncelly*Np+1;
    % the domain is [-Lx, Lx]*[-Ly, Ly]*[-Lz, Lz]
    Lx=1/2; Ly=1/2;
    Param.Ncellx=Ncellx; Param.Ncelly=Ncelly;
    Param.nx = nx; Param.ny = ny;
    fprintf('2D System with total DoFs %d by %d \n',nx,ny);
    fprintf('Laplacian is Q%d spectral element method \n', Np);  
    %x is location ex is 1 or identity and Tx is t and eigx is lambdax in paper
    %or we can understand as diagonal matrix 
    [x,ex,Tx,eigx]=SEGenerator1D('x',Lx,Param);
    [y,ey,Ty,eigy]=SEGenerator1D('y',Ly,Param);
    x=x+1/2;
    y=y+1/2;
    
    TxInv=pinv(Tx); TyInv=pinv(Ty);
    if strcmp(Param.device,'gpu');Device=gpuDevice(Param.deviceID);
       fprintf('GPU computation: starting to load matrices/data \n');
       Tx=gpuArray(Tx); Ty=gpuArray(Ty);
       eigx=gpuArray(eigx);eigy=gpuArray(eigy);
       ex=gpuArray(ex); ey=gpuArray(ey); 
       TxInv=gpuArray(TxInv);TyInv=gpuArray(TyInv); 
    end 
    Lambda2D=squeeze(tensorprod(eigx,ey)+tensorprod(ex,eigy));
    [X_new, Y_new] = meshgrid(x, y);
    coordinates2 = [X(:), Y(:)];
    stepsize=0.1;
end

for sln=1:8
%Start with interpolation of numerical solution
Z=(sol_set(:,:,sln));
Z = interp2(X, Y, Z, X_new, Y_new, 'spline');

for ittnumof=1:repeatnum

scale = 0.01;%

u0x = x;
u0y = y;

%Perturbation for S
perturbation1=0;
for itt1=1:5
for itt2=1:5
u1x=u0x.^(itt1-1);
u1y=u0y.^(itt2-1);
perturbation1=perturbation1+(rand(1)*2-1)*squeeze(u1x*u1y');
end
end
perturbation1=perturbation1/(max(abs(perturbation1(:))));
perturbation1=scale*perturbation1;

%Perturbation for A
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

op=1;
    tic;


    A0 =Z;
    %We only have information for A. We will compute S using A.
    A1 = tensorprod(A0, TyInv', 2, 1);
    A1 = squeeze(tensorprod(TxInv, A1, 2, 1));
    A1 = A1 .* (Lambda2D);
    A1 = tensorprod(A1, Ty', 2, 1);
    A1 = squeeze(tensorprod(Tx, A1, 2, 1));
    nonlA=(mu+rho)*A0-rho;
    A1 = DA*A1+nonlA;
    %Computing S that corresponed to A
    A1 = tensorprod(A1,TyInv',2,1);
    A1 = squeeze(tensorprod(TxInv,A1,2,1));
    A1 = A1./(-DS*Lambda2D -rho);
    A1 = tensorprod(A1,Ty',2,1);
    A1 = squeeze(tensorprod(Tx,A1,2,1));
    Z1=A1;
    %After finding S we will start from numerical solution + perturbation
    S0=A1+perturbation1;S1=S0;
    A1=A0;
    A0=Z+perturbation;

    % Computing the residual part for first equation
    A1 = tensorprod(A0, TyInv', 2, 1);
    A1 = squeeze(tensorprod(TxInv, A1, 2, 1));
    A1 = A1 .* (Lambda2D);
    A1 = tensorprod(A1, Ty', 2, 1);
    A1 = squeeze(tensorprod(Tx, A1, 2, 1));
    nonlA=-S0.*(A0.^2)+(mu+rho)*A0; nonllA=-2*S0.*(A0)+mu+rho;
    A1 = DA*A1+nonlA;

    % Computing the residual part for second equation
    S1 = tensorprod(S0, TyInv', 2, 1);
    S1 = squeeze(tensorprod(TxInv, S1, 2, 1));
    S1 = S1 .* (Lambda2D);
    S1 = tensorprod(S1, Ty', 2, 1);
    S1 = squeeze(tensorprod(Tx, S1, 2, 1));
    nonlS=S0.*(A0.^2)-rho*(1-S0); nonllS=(A0.^2)+rho;
    S1 = DS*S1 +nonlS;

    i=0;
    while i<maxiteration && (norm(A1(:), inf) < maxtol &&  norm(S1(:), inf)<maxtol) && ~any(isnan(A1(:))) && ~any(isnan(S1(:)))
        i=i+1;
        % Computing the residual part
        A1 = tensorprod(A0, TyInv', 2, 1);
        A1 = squeeze(tensorprod(TxInv, A1, 2, 1));
        A1 = A1 .* (Lambda2D);
        A1 = tensorprod(A1, Ty', 2, 1);
        A1 = squeeze(tensorprod(Tx, A1, 2, 1));
        nonlA=-S0.*(A0.^2)+(mu+rho)*A0; nonllA=-2*S0.*(A0)+mu+rho;
        A1 = DA*A1+nonlA;


        S1 = tensorprod(S0, TyInv', 2, 1);
        S1 = squeeze(tensorprod(TxInv, S1, 2, 1));
        S1 = S1 .* (Lambda2D);
        S1 = tensorprod(S1, Ty', 2, 1);
        S1 = squeeze(tensorprod(Tx, S1, 2, 1));
        nonlS=S0.*(A0.^2)-rho*(1-S0); nonllS=(A0.^2)+rho;
        S1 = DS*S1 +nonlS;
        criterian=max(norm(A1(:), inf),norm(S1(:), inf));

	    if abs(criterian)<tol
	    break
	    end

        if op==1

        %Update the beta
		matrix_1U=mu+rho-2*A0(:).*S0(:);
		matrix_2V=rho+A0(:).*A0(:);
		matrix_1V=-A0(:).*A0(:);
		matrix_2U=2*A0(:).*S0(:);
		maax=(matrix_1U+matrix_2V)/2+sqrt(complex((matrix_1U-matrix_2V).^2+4*matrix_1V.*matrix_2U))/2;
        miin=(matrix_1U+matrix_2V)/2-sqrt(complex((matrix_1U-matrix_2V).^2+4*matrix_1V.*matrix_2U))/2;
		betaA=(max(real(maax))+min(real(miin)))/2;
		betaS=betaA;
        end
        %Quasi-Newton %Computing J\F
        A1 = tensorprod(A1,TyInv',2,1);
        A1 = squeeze(tensorprod(TxInv,A1,2,1));
        A1 = A1./(DA*Lambda2D + betaA);
        A1 = tensorprod(A1,Ty',2,1);
        A1 = squeeze(tensorprod(Tx,A1,2,1));
        A0=A0-stepsize*A1;

        S1 = tensorprod(S1,TyInv',2,1);
        S1 = squeeze(tensorprod(TxInv,S1,2,1));
        S1 = S1./(DS*Lambda2D + betaS);
        S1 = tensorprod(S1,Ty',2,1);
        S1 = squeeze(tensorprod(Tx,S1,2,1));
        S0=S0-stepsize*S1;
    end
	elapsed_time = toc;
    fprintf("Our residual is %d after %d th iteration \n",criterian,i)
    

    if norm(A1(:), inf) > tol*100 ||  norm(S1(:), inf) > tol*100
            fprintf("After %d iteration our method diverged.\n", i);
	        break
    elseif norm(Z(:)-A0(:),inf)>0.01
            fprintf("Our method diverged \n");
            break
    end


end

new_sol_set(:,:,sln)=A0;
end







%This is the part that generate all the numerical solutions
for figure_generation=1
cols = 4;
rows = 2;

% Create a tiled layout for better control over subplot spacing
tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% Initialize variables to track global color limits
global_min = Inf;
global_max = -Inf;

% Loop through each subplot
for idx = 1:8
   Z = gather(new_sol_set(:,:, idx));
    nexttile;
    imagesc(Z);

    % Update global color limits
    global_min = min(global_min, min(Z(:)));
    global_max = max(global_max, max(Z(:)));
    axis off
end

% Set global color limits for all subplots
for idx = 1:8
    nexttile(idx);
    caxis([global_min - 0.1, global_max + 0.1]);
end

% Add a single large color bar
cb = colorbar('Position', [0.96 0.11 0.02 0.815]);  % Adjust position to move it more to the right
cb.Label.String = 'Color Scale';

% Adjust figure size
width = 1500;  % Width in pixels
height = 600;  % Height in pixels
set(gcf, 'Position', [100, 100, width, height]);  % [left, bottom, width, height]
end
