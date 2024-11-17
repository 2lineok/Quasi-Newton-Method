
close all
clear all
addpath("Functions")
% Set up the desired GPU device ID

%This part is setting the parameter main code is after this setting.
for parameter_setting=1

    if gpuDeviceCount('available')<1; Param.device='cpu';
    else;Param.device='gpu'; Param.deviceID=1;end % ID=1,2,3,...
    
    
    Np=5;% polynomial degree
    Ncell=40; % number of cells in finite element
    n=Ncell*Np+1;
    
    
    Max_iter=20000;
    
    L=1; % the domain is [-L, L]^2    
    if (Np<2)
        fprintf('polynomial degree Np must be at least 2\n')
        stop
    end
    fprintf('Laplacian is %dth order variational difference (Q%d spectral element) \n', Np+2, Np)  
    fprintf('The order of accuracy for eigenvalues is %d \n', 2*Np)
    Lx=1; Ly=1; Lz=1;
    Param.Ncellx=Ncell; Param.Ncelly=Ncell;
    Param.nx = n; Param.ny = n;
    Param.Np=Np;
    [x,ex,Tx,eigx, H,S,M]=SEGenerator1D_1('x',Lx,Param);
    T = Tx;
    ey=ex;
    ez=ex;
    eigy=eigx;eigz=eigx;Ty=Tx;Tz=Tx;
    invT=inv(T);
    e=ex;
    y=x;
    z=x;
    num_iterations=1;
    b=10;
    a=10;
    uexact = squeeze(tensorprod(cos(x*pi)*e', e)) + squeeze(tensorprod(e*cos(y*pi)', e)) + squeeze(tensorprod(e*e', cos(z*pi)));
    V=squeeze(tensorprod(sin(x*pi/4).^2, e*e'))+squeeze(tensorprod(e,(sin(x*pi/4).^2)*e'))+squeeze(tensorprod(e,e*(sin(x*pi/4).^2)'));
    V=50*V+squeeze(tensorprod(0.5*x.^2, e*e'))+squeeze(tensorprod(e,(0.5*x.^2)*e'))+squeeze(tensorprod(e,e*(0.5*x.^2)'));
    V=2*V;
    f=pi*pi*uexact+a*uexact+b*uexact.^3+V.*uexact;

    if strcmp(Param.device,'gpu');Device=gpuDevice(Param.deviceID);
       fprintf('GPU computation: starting to load matrices/data \n');
       Tx=gpuArray(Tx); Ty=gpuArray(Ty); Tz=gpuArray(Tz);
       eigx=gpuArray(eigx);eigy=gpuArray(eigy);eigz=gpuArray(eigz);
       ex=gpuArray(ex); ey=gpuArray(ey); ez=gpuArray(ez); f=gpuArray(f);
       H=gpuArray(H); V=gpuArray(V); a=gpuArray(a); b=gpuArray(b);
    end 
    Lambda3D=squeeze(tensorprod(eigx,ey*ez')+tensorprod(ex,eigy*ez')...
            +tensorprod(ex,ey*eigz'));
    if strcmp(Param.device,'gpu'); wait(Device);
        fprintf('GPU loading finished and computing started \n');
    end
    scales = [0.1]; %Perturbation scale
end



for scale = scales


for op=1:1
for iter=1:num_iterations  
    %Define perturbation
    u0x = x;
    u0y = y;
    u0z = z;
    perturbation=0;
    for itt1=1:5
        for itt2=1:5
            for itt3=1:5
            u1x=u0x.^(itt1-1);
            u1y=u0y.^(itt2-1);
            u1z=u0z.^(itt3-1);
            perturbation=perturbation+(rand(1)*2-1)*squeeze(tensorprod(u1x*u1y', u1z));
            end
        end
    end
    perturbation=perturbation/(max(abs(perturbation(:))));
    perturbation=scale*perturbation;
    tic;

    old_res=1000;
    res=100;
    u = uexact + perturbation;
    kk=0;

    while (kk<Max_iter & (res<old_res || res>1d-6) && res>10^(-10) && ~isnan(res))  
        kk=kk+1; 
        old_res=res;
        %Computing the residual
        Grad=tensorprod(u,H',3,1)+squeeze(tensorprod(H,u,2,1))+pagemtimes(u,H')+a*u+V.*u+b*u.^3-f;
        %Update the beta      
        if op==1
                beta=(max(max(max(V+3*b*u.^2)))+min(min(min(V+3*b*u.^2))))/2;
        end      
        %Quasi-Newton %Computing J\F      
        res=norm(Grad(:),'inf');
        Grad=tensorprod(Grad,invT',3,1);
        Grad=pagemtimes(Grad,invT');
        Grad=squeeze(tensorprod(invT,Grad,2,1));
        Grad=Grad./(Lambda3D+a+beta);
        Grad=tensorprod(Grad,T',3,1);
        Grad=pagemtimes(Grad,T');
        Grad=squeeze(tensorprod(T,Grad,2,1));
        %Update u
        u=u-Grad;
    end
    elapsed_time = toc;
    if norm(res,inf)>0.00001 | isnan(norm(res,inf))
        fprintf("Our method diverged.")
    end
    fprintf("residual is %d \n",norm(res, inf) );
    fprintf("true error is %d \n", norm(u(:)-uexact(:), inf) );
    fprintf("iteration num is %d \n", kk );

end
    

end
end

