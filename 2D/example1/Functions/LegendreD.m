function [D,r,w] = LegendreD(N)
    Np = N+1; r = JacobiGL(0,0,N);
    w = (2*N+1)/(N*N+N)./power(JacobiP(r,0,0,N),2);   
    Distance = r*ones(1,N+1)-ones(N+1,1)*r'+eye(N+1);
    omega = prod(Distance,2);
    D = diag(omega)*(1./Distance)*diag(1./omega); 
    D(1:Np+1:end) = 0; D(1:Np+1:end) = -sum(D,2);
end
function [x] = JacobiGL(alpha,beta,N)
    x = zeros(N+1,1);
    if (N==1); x(1)=-1.0; x(2)=1.0; return; end
    [xint,temp] = JacobiGQ(alpha+1,beta+1,N-2);
    x = [-1, xint', 1]'; return;
end
function [x,w] = JacobiGQ(alpha,beta,N)
    if (N==0) 
        x(1) = -(alpha-beta)/(alpha+beta+2); w(1) = 2; return; 
    end
    h1 = 2*(0:N)+alpha+beta;
    J = diag(-1/2*(alpha*alpha-beta*beta)./(h1+2)./h1) + ...
        diag(2./(h1(1:N)+2).*sqrt((1:N).*((1:N)+alpha+beta).*...
        ((1:N)+alpha).*((1:N)+beta)./(h1(1:N)+1)./(h1(1:N)+3)),1);
    if (alpha+beta<10*eps); J(1,1)=0.0; end
    J = J + J'; [V,D] = eig(J); x = diag(D);
    w = power(V(1,:)',2)*power(2,alpha+beta+1)/(alpha+beta+1)*...
        gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1);
end
function [P] = JacobiP(x,alpha,beta,N)
    xp = x; dims = size(xp);
    if (dims(2)==1); xp = xp'; end    
    PL = zeros(N+1,length(xp));   
    gamma0 = power(2,alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*...
        gamma(beta+1)/gamma(alpha+beta+1);
    PL(1,:) = 1.0/sqrt(gamma0);
    if (N==0); P = PL'; return; end
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0;
    PL(2,:) = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/sqrt(gamma1);
    if (N==1); P = PL(N+1,:)'; return; end
    aold = 2/(2+alpha+beta)*sqrt((alpha+1)*(beta+1)/(alpha+beta+3));    
    for i = 1:N-1
      h1 = 2*i+alpha+beta;
      anew = 2/(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*...
          (i+1+beta)/(h1+1)/(h1+3));
      bnew = - (alpha*alpha-beta*beta)/h1/(h1+2);
      PL(i+2,:) = 1/anew*( -aold*PL(i,:) + (xp-bnew).*PL(i+1,:));
      aold = anew;
    end  
    P = PL(N+1,:)';
end