function [varargout] = SEGenerator1D_1(direction,L,Param)
% generate 1D spectral element with Neumann B.C.
switch direction
    case 'x'
        N=Param.Np; Ncell=Param.Ncellx; n=Param.nx;
    case 'y'
        N=Param.Np; Ncell=Param.Ncelly; n=Param.ny;
    case 'z'
        N=Param.Np; Ncell=Param.Ncellz; n=Param.nz;
end
% generate the mesh with Ncell intervals with domain [Left, Right]
[D,r,w] = LegendreD(N); Left = -L; Right = L;
Length = Right - Left; dx = Length/Ncell;
for j = 1:Ncell
    cellLeft = Left+dx*(j-1);
    localPoints = cellLeft+dx/2+r*dx/2;
    if (j==1)
    x = localPoints;
    else
    x = [x;localPoints(2:end)];
    end
end        
SLocal = D'*diag(w)*D; % local stiffness matrix for each element       
S=[]; M=[];
for j = 1:Ncell % global stiffness and lumped mass matrices
    S = blkdiag(SLocal,S); M = blkdiag(diag(w),M); 
end
% Next step: "glue" the cells
Np = N+1; % number of points in each cell
Glue = sparse(zeros(Ncell*Np-Ncell+1, Ncell*Np));
for j = 1:Ncell
    rowStart=(j-1)*Np+2-j; rowEnd=rowStart+Np-1;
    colStart=(j-1)*Np+1; colEnd=colStart+Np-1;
    Glue(rowStart:rowEnd,colStart:colEnd)=speye(Np); 
end
S=Glue*S*Glue'; M=Glue*M*Glue'; H=diag(1./diag(M))*S; 
ex=ones(n,1); MHalfInv=diag(1./sqrt(diag(M)));
S1=MHalfInv*S*MHalfInv; S1=(S1+S1')/2;
[U,d]=eig(S1,'vector'); [lambda,indexSort]=sort(d);
T=U(:,indexSort); h=dx/2; lambda=lambda/(h*h); 
S1=sparse(S1/(h*h)); M=sparse(M);
% after this step, T is the eigenvector of H
T = MHalfInv*T; H=full(H/(h*h)); S=S/h; M=full(M*h);
varargout{1}=x; varargout{2}=ex; varargout{3}=T; varargout{4}=lambda; varargout{5}=H; varargout{6}=S; varargout{7}=M;
end
