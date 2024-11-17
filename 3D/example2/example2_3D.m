
clear all
close all

load('numerical_solution.mat')

u0=gather(u);
alpha=1600;



BC='Dirichlet';% BC='NeumannBC';% BC='Periodic_';
L=8;
Left=-L;
Right=L;
[nn, x,dx, H, M, S,T,invT,lambda]= SEGenerator1D_2(Left,Right,Ncell, Np, BC);
y=x;z=x;


M3D=squeeze(tensorprod(diag(M)*diag(M)', diag(M)));

e=ones(length(x),1);

Lambda3D=squeeze(tensorprod(lambda, e*e'))+squeeze(tensorprod(e,lambda*e'))+squeeze(tensorprod(e,e*lambda'));
u=squeeze(tensorprod(e, e*e'));
V=squeeze(tensorprod(sin(x*pi/4).^2, e*e'))+squeeze(tensorprod(e,(sin(x*pi/4).^2)*e'))+squeeze(tensorprod(e,e*(sin(x*pi/4).^2)'));
V=50*V+squeeze(tensorprod(0.5*x.^2, e*e'))+squeeze(tensorprod(e,(0.5*x.^2)*e'))+squeeze(tensorprod(e,e*(0.5*x.^2)'));
V=2*V;





stepsize=1;
scales = [0.3];
num_iterations=1;




fprintf("This is start \n")
for scale = scales




for op=1

for iter=1:num_iterations
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
    u=u0+perturbation;
    u=u/sqrt(sum(u.*M3D.*u, 'all'));
    old_res=1000;
    res=100;
    kk=0;
    tic
    while (kk<20000 & (res<old_res | res>1d-6))  
        if res<10^(-10)
            break
        end
        kk=kk+1; 
        old_res=res;
        
        if op==1
            beta=(max(max(max(V+3*alpha*u.^2)))+min(min(min(V+3*alpha*u.^2))))/2;
        end      
                
       Grad=tensorprod(u,H',3,1)+squeeze(tensorprod(H,u,2,1))+pagemtimes(u,H')+V.*u+alpha*u.^3;
       somecon=sum(M3D.*Grad.*u, 'all');
       res=norm(u(:)-Grad(:)/sqrt(sum(Grad.*M3D.*Grad, 'all')))/norm(u(:));
       Grad=Grad-somecon*u;
       Grad=tensorprod(Grad,invT',3,1);
       Grad=pagemtimes(Grad,invT');
       Grad=squeeze(tensorprod(invT,Grad,2,1));
       Grad=Grad./(Lambda3D+beta-somecon);
       Grad=tensorprod(Grad,T',3,1);
       Grad=pagemtimes(Grad,T');
       Grad=squeeze(tensorprod(T,Grad,2,1));

       u=u-stepsize*Grad;
       u=u/sqrt(sum(u.*M3D.*u, 'all'));
        
        fprintf('Iteration of our method is %d and has residue %d \n', kk, res)
 
    end

    elapsed_time = toc;

	if norm(res, inf)<10^(-7)
        fprintf("our scale is %d \n",scale);
	    fprintf("this is error %d \n",norm(u(:)-u0(:), inf) );
        fprintf('Iteration %d has residue %d when degree of freedom is %d \n', kk, res,Ncell*Np)
    else
        fprintf("Our method did not converged properly")
	end

end


end

end






[X, Y, Z] = meshgrid(x,y,z);
u=gather(u);
for mod=1:2
if mod==1

figure;

slice(X,Y,Z,u,[0],[0],[0]);

xlabel('X');
ylabel('Y');
zlabel('Z');
title('Slice Visualization');
colorbar;
shading flat
axis equal;

clim([min(u(:)) max(u(:))]);
view(45,18)

grid off; 
else

    FigHandle = figure;
  
     
    set(FigHandle, 'Position', [100, 100, 800, 800]);
    set(0,'DefaultTextFontSize',15,'DefaultAxesFontSize', 15)
 
    isosurface(X,Y,Z,u,0.002);
    axis equal;
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    xlim([-8 8])
    ylim([-8 8])
    zlim([-8 8])
    view(57,10) 
    FigHandle = figure;


    set(FigHandle, 'Position', [200, 200, 800, 800]);
    set(0,'DefaultTextFontSize',15,'DefaultAxesFontSize', 15)

    slice(X,Y,Z,u,[],[],[-4 0 4]);
    shading flat
    xlabel('X')
    ylabel('Y')
    zlabel('Z')

    
    axis tight;axis equal; colorbar
   
    xlim([-8 8])
    ylim([-8 8])
    zlim([-8 8])
    view(45,18)
    colormap(jet) 
end
end