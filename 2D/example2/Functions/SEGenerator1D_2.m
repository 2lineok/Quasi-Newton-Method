function [n, x,dx, H, M, S,T,invT,lambda]= SEGenerator1D_2(Left,Right,Ncell, N, BC);
%% generate matrices and grids for P^N spectral element CG with Neumann, Dirichlet, Periodic b.c.
%% for -u''=f

% Xiangxiong Zhang, Purdue University, 2023

% inputs:
% BC='NeumannBC'; 
% BC='Periodic_';
% BC='Dirichlet';

% outputs: M*(dx/2) is the mass matrix; S/(dx/2) is the stiffness matrix
%          H/(dx/2)^2=Mass^{-1}*Stiffness



% Compute basic Legendre Gauss Lobatto grid, weights and differentiation
% matrix D where D_{ij}=l'_j(r_i) and l_j(r) is the Lagrange polynomial.
[D,r,w]=LegendreD(N);

% Generate the mesh with Ncell intervals and each interval has GL points
% Domain is [Let, Right]

Length=Right-Left;
dx=Length/Ncell;
for j=1:Ncell
    cell_left=Left+dx*(j-1);
    local_points=cell_left+dx/2+r*dx/2;
    if (j==1)
        x=local_points;
    else
        x=[x;local_points(2:end)];
    end
end

S_local=D'*diag(w)*D; % local stiffness matrix for each element

S=[];
M=[];
for j=1:Ncell
    S=blkdiag(S_local,S); % global stiffness and lumped mass matrices if treating cells sperately
    M=blkdiag(diag(w),M); 
end

% Next step: "glue" the cells
Np=N+1; % number of points in each cell
Glue=sparse(zeros(Ncell*Np-Ncell+1, Ncell*Np));
for j=1:Ncell
    rowstart_index=(j-1)*Np+2-j;
    rowend_index=rowstart_index+Np-1;
    colstart_index=(j-1)*Np+1;
    colend_index=colstart_index+Np-1;
    Glue(rowstart_index:rowend_index,colstart_index:colend_index)=speye(Np); 
end
if (BC=='Periodic_')
    Glue(1,end)=1;
    Glue(end,1)=1;
end
S=Glue*S*Glue';
M=Glue*M*Glue';
H=diag(1./diag(M))*S;

if (BC=='Periodic_')
    S=S(1:end-1, 1:end-1);
    M=M(1:end-1, 1:end-1);
    H=H(1:end-1, 1:end-1);   
    x=x(1:end-1);
end


if (BC=='Dirichlet')
    S=S(2:end-1, 1:end);
    M=M(2:end-1, 1:end);
    H=diag(1./diag(M(:,2:end-1)))*S(:,2:end-1);
    x=x(2:end-1);
end
if (BC=='Dirichlet')
  S=S(:,2:end-1);
  M=M(:,2:end-1);
end

M_half_inv=diag(1./sqrt(diag(M)));
S1=M_half_inv*S*M_half_inv;    
S1=(S1+S1')/2;
[U,d]=eig(S1,'vector');
[lambda, ind] = sort(d);
T = U(:, ind);
h=dx/2; 
lambda=lambda/(h)^2; 
M=sparse(M);
size(M)
% after this step, T is the eigenvector of H
T=M_half_inv*T;
invT=inv(T); % T does not have orthogonal columns because H is not symmetric

H=full(H/h^2);
S=S/h;
M=full(M*h);

    n=numel(x);
end