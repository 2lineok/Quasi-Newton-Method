clear all

load("all.mat")

newSEMsol=newSEMsol(:,[4,7,9,11,23,26,29,30]);
repeatnum=1;
if gpuDeviceCount('available')<1; Param.device='cpu';
else;Param.device='gpu'; Param.deviceID=1;end % ID=1,2,3,...
Np=5; Param.Np=Np; % polynomial degree Q5
Ncellx=20; Ncelly=20; % finite element cell number
% total number of unknowns in each direction
nx=Ncellx*Np+1; ny=Ncelly*Np+1;
% the domain is [-Lx, Lx]*[-Ly, Ly]*[-Lz, Lz]
Lx=1/2; Ly=1/2; DA=2.5*10^(-4); DS=5*10^(-4); mu=0.065; rho=0.04;maxiteration=30000;tol=10^(-9); maxtol=1000;
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
[X, Y] = meshgrid(x, y);




if gpuDeviceCount('available')<1; Param.device='cpu';
else;Param.device='gpu';end % ID=1,2,3,...
Np=5; Param.Np=Np; % polynomial degree Q5
Ncellx=10; Ncelly=10; % finite element cell number
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

for sln=1:8%:size(newSEMsol,2) %1:size(newSEMsol,2)
totalnumcon=0;
new_solution=newSEMsol(:,sln);
Z = reshape(new_solution, sqrt(length(new_solution)), sqrt(length(new_solution)));
Z = interp2(X, Y, Z, X_new, Y_new, 'spline');
iterations = zeros(repeatnum, 1);
times = zeros(repeatnum, 1);
results = struct('Method', {}, 'Connum', {}, 'Time', {}, 'Iterations', {});


for ittnumof=1:repeatnum
pltrue=0;

if norm(new_solution,'fro')<0.001
    continue
end

% Display the interpolated solution
% Reshape the solution into a matrix



stepsize=0.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scale = 0.01;%, 0.02, 0.05 ,01

u0x = x;
u0y = y;

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
    pltrue=0;
    tic;
    fprintf("this is op is %d and iteration is %d and solution is %d \n", op,ittnumof,sln);


    A0 =Z;

    A1 = tensorprod(A0, TyInv', 2, 1);
    A1 = squeeze(tensorprod(TxInv, A1, 2, 1));
    A1 = A1 .* (Lambda2D);
    A1 = tensorprod(A1, Ty', 2, 1);
    A1 = squeeze(tensorprod(Tx, A1, 2, 1));
    nonlA=(mu+rho)*A0-rho;
    A1 = DA*A1+nonlA;
    
    A1 = tensorprod(A1,TyInv',2,1);
    A1 = squeeze(tensorprod(TxInv,A1,2,1));
    A1 = A1./(-DS*Lambda2D -rho);
    A1 = tensorprod(A1,Ty',2,1);
    A1 = squeeze(tensorprod(Tx,A1,2,1));
    Z1=A1;
    S0=A1+perturbation1;S1=S0;
    A1=A0;
    A0=Z+perturbation;
    if op==1 && ittnumof == 1
        figure;
        subplot(1, 2, 1);
        width = 1500;  % Width in pixels
        height = 600;  % Height in pixels
        set(gcf, 'Position', [100, 100, width, height]);  % [left, bottom, width, height]
        imagesc(A0);
        colorbar;
        ax = gca;  % Get current axes
        if min(A0(:)) ~= max(A0(:))
            ax.CLim = [gather(min(A0(:)))-0.1 gather(max(A0(:)))+0.1];
        end 
        title('A0');
        
        % Create the second subplot (1 row, 2 columns, second subplot)
        subplot(1, 2, 2);
        imagesc(S0);
        colorbar;
        ax = gca;  % Get current axes
        if min(S0(:)) ~= max(S0(:))
            ax.CLim = [gather(min(S0(:)))-0.1 gather(max(S0(:)))+0.1]; 
        end
        title('S0 ');
        % Define the directory name
    end


    i=0;
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
	if i==1
	fprintf("Our residual is %d",criterian)
	end
	if abs(criterian)<tol
	fprintf("Our residual is %d after %d th iteration",criterian,i)
	break
	end
		stepsize=0.1;
        % Online computation %do inverse of f
        if op==1
		jmatrixA=mu+rho-2*A0(:).*S0(:);
		jmatrixB=rho+A0(:).*A0(:);
		jj=-(A0(:).*A0(:));
		ji=(2*A0(:).*S0(:));
		maax=(jmatrixA+jmatrixB)/2+sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
        miin=(jmatrixA+jmatrixB)/2-sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
		betaA=(max(real(maax))+min(real(miin)))/2;
        %betaA=5;
		betaS=betaA;

        elseif op==2
		jmatrixA=mu+rho-2*A0(:).*S0(:);
		jmatrixB=rho+A0(:).*A0(:);
		jj=-(A0(:).*A0(:));
		ji=(2*A0(:).*S0(:));
		maax=(jmatrixA+jmatrixB)/2+sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
        miin=(jmatrixA+jmatrixB)/2-sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
		betaA=(max(real(maax))+min(real(miin)));
		betaS=betaA;

        elseif op==3
		jmatrixA=mu+rho-2*A0(:).*S0(:);
		jmatrixB=rho+A0(:).*A0(:);
		jj=-(A0(:).*A0(:));
		ji=(2*A0(:).*S0(:));
		maax=(jmatrixA+jmatrixB)/2+sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
        miin=(jmatrixA+jmatrixB)/2-sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
		betaA=(min(real(miin))+min(real(miin)))/2;
		betaS=betaA;

        elseif op==4
		jmatrixA=mu+rho-2*A0(:).*S0(:);
		jmatrixB=rho+A0(:).*A0(:);
		jj=-(A0(:).*A0(:));
		ji=(2*A0(:).*S0(:));
		maax=(jmatrixA+jmatrixB)/2+sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
        miin=(jmatrixA+jmatrixB)/2-sqrt(complex((jmatrixA-jmatrixB).^2/4-2*A0(:).^3.*S0(:)));
		betaA=(max(real(maax))+max(real(maax)))/2;
		betaS=betaA;

        elseif op==5
                betaA = 0.1;
                betaS = betaA;
        elseif op==6
                betaA = 0.5;
                betaS = betaA;
        elseif op==7
                betaA = 1;
                betaS = betaA;
        elseif op==8
                betaA = 5;
                betaS = betaA;
        elseif op==9
                betaA = 10;
                betaS = betaA;
        elseif op==10
                betaA = 20;
                betaS = betaA;
        else
                error('Invalid op value');
        end
        % online computation %do inverse of f
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
    
    

    if norm(A1(:), inf) > tol*100 ||  norm(S1(:), inf) > tol*100
            fprintf("this is after i %d iteration and did not converged \n", i);
            fprintf("this is betaA is %f and betaS is %f \n", betaA, betaS);
	        break
    elseif norm(Z(:)-A0(:),inf)>0.01
            fprintf("nonononononono %f and betaS is %f \n", betaA, betaS);
            pltrue=2;
            break
    else
        pltrue=1;
        totalnumcon=1+totalnumcon;
    end


end

sol_set(:,:,sln)=A0;
end















cols = 4;
rows = 2;

% Create a tiled layout for better control over subplot spacing
tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% Initialize variables to track global color limits
global_min = Inf;
global_max = -Inf;

% Loop through each subplot
for idx = 1:8
   Z = gather(sol_set(:,:, idx));
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


% Add a single large horizontal color bar
%cb = colorbar('Position', [0.1 0.005 0.8 0.05], 'Orientation', 'horizontal');  % Adjust position and size
%cb.Label.String = 'Color Scale';



% Adjust figure size
width = 1500;  % Width in pixels
height = 600;  % Height in pixels
set(gcf, 'Position', [100, 100, width, height]);  % [left, bottom, width, height]
save("all1.mat","sol_set","-v7.3")
