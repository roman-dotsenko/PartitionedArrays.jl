function [x,flag,iter,Miter,QLPiter,relres,relAres,...
          Anorm,Acond,xnorm,Axnorm,resvec,Aresvec] = ...
	minresQLP(A,b,rtol,maxit,M,shift,maxxnorm,Acondlim,TranCond,show)
%minresQLP: min-length solution to symmetric (possibly singular) Ax=b or min||Ax-b||.
%
%   X = minresQLP(A,B) solves the system of linear equations A*X=B
%   or the least-squares problem min norm(B-A*X) if A is singular.
%   The N-by-N matrix A must be symmetric or Hermitian, but need not be
%   positive definite or nonsingular.  It may be double or single.
%   The rhs vector B must have length N.  It may be real or complex,
%   double or single, 
%
%   X = minresQLP(AFUN,B) accepts a function handle AFUN instead of
%   the matrix A.  Y = AFUN(X) returns the matrix-vector product Y=A*X.
%   In all of the following syntaxes, A can be replaced by AFUN.
%
%   X = minresQLP(A,B,RTOL) specifies a stopping tolerance.
%   If RTOL=[] or is absent, a default value is used.
%   (Similarly for all later input parameters.)
%   Default RTOL=1e-6.
%
%   X = minresQLP(A,B,RTOL,MAXIT)
%   specifies the maximum number of iterations.  Default MAXIT=N.
%
%   X = minresQLP(A,B,RTOL,MAXIT,M)
%   uses a matrix M as preconditioner.  M must be positive definite
%   and symmetric or Hermitian.  It may be a function handle MFUN
%   such that Y=MFUN(X) returns Y=M\X.
%   If M=[], a preconditioner is not applied.
%
%   X = minresQLP(A,B,RTOL,MAXIT,M,SHIFT)
%   solves (A - SHIFT*I)X = B, or the corresponding least-squares problem
%   if (A - SHIFT*I) is singular, where SHIFT is a real or complex scalar.
%   Default SHIFT=0.
%
%   X = minresQLP(A,B,RTOL,MAXIT,M,SHIFT,MAXXNORM,ACONDLIM,TRANCOND)
%   specifies three parameters associated with singular or
%   ill-conditioned systems (A - SHIFT*I)*X = B.
%
%   MAXXNORM is an upper bound on NORM(X).
%   Default MAXXNORM=1e7.
%
%   ACONDLIM is an upper bound on ACOND, an estimate of COND(A).
%   Default ACONDLIM=1e15.
%
%   TRANCOND is a real scalar >= 1.
%   If TRANCOND>1,        a switch is made from MINRES iterations to
%                         MINRES-QLP iterationsd when ACOND >= TRANCOND.
%   If TRANCOND=1,        all iterations will be MINRES-QLP iterations.
%   If TRANCOND=ACONDLIM, all iterations will be conventional MINRES
%                         iterations (which are slightly cheaper).
%   Default TRANCOND=1e7.
%
%   X = minresQLP(A,B,RTOL,MAXIT,M,SHIFT,MAXXNORM,ACONDLIM,TRANCOND,SHOW)
%   specifies the printing option.
%   If SHOW=true,  an iteration log will be output.
%   If SHOW=false, the log is suppressed.
%   Default SHOW=true.
%   
%
%   [X,FLAG] = minresQLP(A,B,...) returns a convergence FLAG:
%   -1 (beta2=0)  B and X are eigenvectors of (A - SHIFT*I).
%    0 (beta1=0)  B = 0.  The exact solution is X = 0.                    
%    1 X solves the compatible (possibly singular) system (A - SHIFT*I)X = B
%      to the desired tolerance:
%         RELRES = RNORM / (ANORM*XNORM + NORM(B)) <= RTOL,
%      where
%              R = B - (A - SHIFT*I)X and RNORM = norm(R).
%    2 X solves the incompatible (singular) system (A - SHIFT*I)X = B
%      to the desired tolerance:
%         RELARES = ARNORM / (ANORM * RNORM) <= RTOL,
%      where
%              AR = (A - SHIFT*I)R and ARNORM = NORM(AR).
%    3 Same as 1 with RTOL = EPS.
%    4 Same as 2 with RTOL = EPS.
%    5 X converged to an eigenvector of (A - SHIFT*I).
%    6 XNORM exceeded MAXXNORM.
%    7 ACOND exceeded ACONDLIM.
%    8 MAXIT iterations were performed before one of the previous
%      conditions was satisfied.
%    9 The system appears to be exactly singular.  XNORM does not
%      yet exceed MAXXNORM, but would if further iterations were
%      performed.
%
%    [X,FLAG,ITER,MITER,QLPITER] = minresQLP(A,B,...) returns the
%    number of iterations performed, with ITER = MITER + QLPITER.
%    MITER   is the number of conventional MINRES iterations.
%    QLPITER is the number of MINRES-QLP iterations.
%
%    [X,FLAG,ITER,MITER,QLPITER,RELRES,RELARES] = minresQLP(A,B,...)
%    returns relative residuals for (A - SHIFT*I)X = B and the
%    associated least-squares problem.  RELRES and RELARES are
%    defined above in the description of FLAG.
%
%    [X,FLAG,ITER,MITER,QLPITER,RELRES,RELARES,ANORM,ACOND,XNORM,AXNORM] =
%       minresQLP(A,B,...) returns
%       ANORM,  an estimate of the 2-norm of A-SHIFT*I.
%       ACOND,  an estimate of COND(A-SHIFT*I,2).
%       XNORM,  a recurred estimate of NORM(X).
%       AXNORM, a recurred estimate of NORM((A-SHIFT*I)X)
%
%    [X,FLAG,ITER,MITER,QLPITER,RELRES,RELARES,ANORM,ACOND,XNORM,AXNORM,...
%       RESVEC,ARESVEC] = minresQLP(A,B,...) returns
%       RESVEC,  a vector of estimates of NORM(R) at each iteration,
%                including NORM(B) as the first entry.
%       ARESVEC, a vector of estimates of NORM((A-SHIFT*I)R) at each
%                iteration, including NORM((A-SHIFT*I)B) as the first entry.
%       RESVEC and ARESVEC have length ITER+1.
%    
%
% EXAMPLE 1: 
%   n = 100;                                  e = ones(n,1);
%   A = spdiags([-2*e 4*e -2*e],-1:1,n,n);    M = spdiags(4*e,0,n,n);
%   b = sum(A,2);             rtol = 1e-10;   maxit = 50;
%   x = minresQLP(A,b,rtol,maxit,M);
%   
%   Alternatively, use this matrix-vector product function:
%     function y = Afun(x,n)
%       y = 4*x;
%       y(2:n)   = y(2:n)   - 2*x(1:n-1);
%       y(1:n-1) = y(1:n-1) - 2*x(2:n);
%   as input to minresQLP:
%   n = 100;
%   A = @(x)Afun(x,n);
%   x = minresQLP(A,b,rtol,maxit,M);
%
% EXAMPLE 2: A is Laplacian on a 50 by 50 grid, singular and indefinite.
%   n = 50;  N = n^2;  e = ones(n,1);
%   B = spdiags([e e e], -1:1, n, n);
%   A = sparse([],[],[],N,N,(3*n-2)^2);
%   for i=1:n 
%     A((i-1)*n+1:i*n,(i-1)*n+1:i*n) = B; 
%     if i*n+1 < n*n,   A(i*n+1:(i+1)*n,(i-1)*n+1:i*n)     = B; end
%     if (i-2)*n+1 > 0, A((i-2)*n+1:(i-1)*n,(i-1)*n+1:i*n) = B; end
%   end
%   b = sum(A,2);   rtol   = 1e-5;    shift = 0;    maxxnorm = 1e2; 
%   M = [];         Acondlim = [];    show  = true;
%   x = minresQLP(A,b,rtol,N,M,shift,maxxnorm,Acondlim,show);
%
% EXAMPLE 3: A is diagonal, singular and indefinite. 
%   h = 1;  a = -10; b = -a; n = 2*b/h + 1;   
%   A = spdiags((a:h:b)', 0, n, n);
%   b = ones(n,1);  rtol   = 1e-6;    shift = 0;    maxxnorm = 1e2; 
%   M = [];         Acondlim = [];    show  = true;
%   x = minresQLP(A,b,rtol,N,M,shift,maxxnorm,Acondlim,show);
%
% See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, PCG, QMR, SYMMLQ,
% TFQMR, CHOLINC, FUNCTION_HANDLE.
% Also MINRES, SYMMLQ, LSQR, CGLS downloadable from
%   http://www.stanford.edu/group/SOL/software.html
% 
% REFERENCES:
% S.-C. Choi, C. C. Paige, and M. A. Saunders, 
% MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems,
% SIAM Journal of Scientific Computing, submitted on March 7, 2010.
%
% S.-C. Choi's PhD Dissertation, Stanford University, 2006:
%   http://www.stanford.edu/group/SOL/dissertations.html
%    
% CURRENT / FUTURE RELEASES of minresQLP:
%   http://www.stanford.edu/group/SOL/software.html


% MODIFICATION HISTORY:
%   12 May 2006: Created MINRESQLPs.m from research file minresQLP35.m. 
%   19 Apr 2010: Help formatted like Matlab routines. Added output Aresvec.
%   25 Apr 2010: Eliminated unreferenced variables. Shortened
%                resvec and Aresvec from maxit to (iter+1).
%                The final rnorm, Arnorm, xnorm are computed directly.
%   26 Apr 2010: Special tests for beta2=0 and/or alfa1=0 when no preconditioning.
%   02 May 2010: First version for http://www.stanford.edu/group/SOL/software.html.
%
% KNOWN BUGS:
%   DD Mmm YYYY: ---
%
% NOTES:
%   DD Mmm YYYY: ---
%
% AUTHORS: Sou-Cheng (Terrya) Choi, iCME, Stanford University
%          Chris Paige, Computer Science Dept, McGill University
%          Michael Saunders, SOL, Stanford University
%
% COPYRIGHT NOTICE:
%
%   This is Copyrighted Material. The software is COPYRIGHTED by the  
%   original authors.
% 
% COPYLEFT NOTICE:
%
%   Permission is granted to make and distribute verbatim copies of this 
%   file, provided that the entire file is copied **together** as a 
%   **unit**.
% 
%   The purpose of this permission notice is to allow you to make copies 
%   of the software and distribute them to others, for free or for a fee, 
%   subject to the constraint that you maintain the entire file here 
%   as a unit.  This enables people you give the software to be aware 
%   of its origins, to ask questions of us by e-mail, to request  
%   improvements, obtain later releases, and so forth.
% 
%   If you seek permission to copy and distribute translations of this 
%   software into another language, please e-mail a specific request to
%   saunders@stanford.edu and scchoi@stanford.edu.
% 
%   If you seek permission to excerpt a **part** of the software library,
%   for example to appear in a scientific publication, please e-mail a
%   specific request to saunders@stanford.edu and scchoi@stanford.edu.
% 
% COMMENTS? 
%
%   Email saunders@stanford.edu and scchoi@stanford.edu


%%  Check inputs and set default values.
n      = length(b);
precon = true;
if nargin <  2,                             error('Not enough input parameters');  end
if nargin <  3 || ~exist('rtol'    ,'var') || isempty(rtol)    , rtol     = 1e-6;  end
if nargin <  4 || ~exist('maxit'   ,'var') || isempty(maxit)   , maxit    = n;     end
if nargin <  5 || ~exist('M'       ,'var') || isempty(M)       , precon   = false; end
if nargin <  6 || ~exist('shift'   ,'var') || isempty(shift)   , shift    = 0;     end
if nargin <  7 || ~exist('maxxnorm','var') || isempty(maxxnorm), maxxnorm = 1e7;   end
if nargin <  8 || ~exist('Acondlim','var') || isempty(Acondlim), Acondlim = 1e15;  end
if nargin <  9 || ~exist('TranCond','var') || isempty(TranCond), TranCond = 1e7;   end
if nargin < 10 || ~exist('show'    ,'var') || isempty(show)    , show     = true;  end
%if nargin< 11 || ~exist('disable' ,'var') || isempty(disable) , disable  = false; end

if nargout> 11
  resvec  = zeros(maxit+1,1);
  Aresvec = zeros(maxit+1,1);
else
  resvec  = [];
  Aresvec = [];
end

%% Set up {beta1, p, v} for the first Lanczos vector v1.
r2    = full(b);    % r2    = b
r3    = r2;         % r3    = b
beta1 = norm(r2);   % beta1 = norm(b)
if precon
  r3    = minresxxxM(M,r2);   % M*r3  = b
  beta1 = r3'*r2;             % beta1 = b'*inv(M)*b
  if beta1 < 0
    error('"M" appears to be indefinite.');
  else
    beta1 = sqrt(beta1);     
  end
end

%% Initialize other quantities.
flag0    = -2;     flag     = flag0;
iter     = 0;      QLPiter  = 0;
lines    = 1;      headlines= 20;
beta     = 0;      tau      = 0;          taul     = 0;      phi      = beta1;
betan    = beta1;  gmin     = 0;          cs       = -1;     sn       = 0;
cr1      = -1;     sr1      = 0;          cr2      = -1;     sr2      = 0;
dltan    = 0;      eplnn    = 0;          gama     = 0;      gamal    = 0;
gamal2   = 0;      eta      = 0;          etal     = 0;      etal2    = 0;
vepln    = 0;      veplnl   = 0;          veplnl2  = 0;      ul3      = 0;
ul2      = 0;      ul       = 0;          u        = 0;      rnorm    = betan;
xnorm    = 0;      xl2norm  = 0;          Axnorm   = 0;
Anorm    = 0;      Acond    = 1;
relres   = rnorm / (beta1 + 1e-50);       % Safeguard for beta1 = 0
x        = zeros(n,1);
w        = zeros(n,1);
wl       = zeros(n,1);

if ~isempty(resvec)
  resvec(1)  = beta1;
end

%% print header if show
first = 'Enter minresQLP.  ';
last  = 'Exit minresQLP.  ';  
msg=[' beta2 = 0.  b and x are eigenvectors                   '   % -1
     ' beta1 = 0.  The exact solution is  x = 0               '   %  0
     ' A solution to Ax = b found, given rtol                 '   %  1
     ' Min-length solution for singular LS problem, given rtol'   %  2
     ' A solution to Ax = b found, given eps                  '   %  3
     ' Min-length solution for singular LS problem, given eps '   %  4
     ' x has converged to an eigenvector                      '   %  5
     ' xnorm has exceeded maxxnorm                            '   %  6
     ' Acond has exceeded Acondlim                            '   %  7
     ' The iteration limit was reached                        '   %  8
     ' Least-squares problem but no converged solution yet    ']; %  9 
if show 
  fprintf('\n%s%s', first, 'Min-length solution of symmetric (A-sI)x = b or min ||(A-sI)x - b||')
  fprintf('\nn      =%7g   ||b||    =%10.3e   shift    =%10.3e   rtol     =%10.3e',...
             n,            beta1,             shift,             rtol)  
  fprintf('\nmaxit  =%7g   maxxnorm =%10.3e   Acondlim =%10.3e   TranCond =%10.3e',...
             maxit,        maxxnorm,          Acondlim,          TranCond) 
  fprintf('\nprecon =%7g\n' , precon)
  head = '    iter     rnorm     Arnorm   Compatible     LS        Anorm      Acond      xnorm';
  fprintf('\n%s\n', head)
end

if beta1==0, flag = 0; end   % b = 0 => x = 0.  We will skip the main loop.


%% Main iteration 
while flag == flag0 && iter < maxit

%% Lanczos
  iter  = iter + 1;
  betal = beta;        beta = betan;     
  v = r3*(1/beta);     r3 = minresxxxA(A,v);  
  if shift ~= 0,       r3 = r3 - shift*v;         end
  if iter  >  1,       r3 = r3 - (beta/betal)*r1; end

  alfa = real(r3'*v);  %%%%% Allow for Hermitian A.  Must get real alfa here.
  r3   = r3 - (alfa/beta)*r2;     r1 = r2;     r2 = r3;                  

  if ~precon
    betan = norm(r3);                  
    if iter == 1       % Something special can happen
      if betan == 0    % beta2 = 0
	if alfa == 0     % alfa1 = 0
	  flag = 0;      % Ab = 0 and x = 0         ("A" = (A - shift*I))
          break
	else
	  flag = -1;     % Ab = alfa1 b,  x = b/alfa1, an eigenvector
          x    = full(b)/alfa;
	  break
	end
      end
    end
  else
    r3 = minresxxxM(M,r2);     betan = r2'*r3; 
    if betan > 0
      betan = sqrt(betan);     
    else
      error('"M" appears to be indefinite or singular.');     
    end
  end
  pnorm  = norm([betal alfa betan]);

%% Apply previous left rotation Q_{k-1} 
  dbar  = dltan;
  dlta  = cs*dbar + sn*alfa;    epln     = eplnn;
  gbar  = sn*dbar - cs*alfa;    eplnn    = sn*betan;
  dltan = -cs*betan;            dlta_QLP = dlta;     

%% Compute the current left plane rotation Q_k
  gamal3 = gamal2;     gamal2 = gamal;     gamal    = gama;     
  [cs,sn,gama] = SymGivens2(gbar, betan);  gama_tmp = gama;
  taul2  = taul;       taul   = tau;       tau      = cs*phi;     
  Axnorm = norm([Axnorm tau]);             phi      = sn*phi;    

%% Apply the previous right plane rotation P{k-2,k}
  if iter > 2     
    veplnl2  = veplnl;     etal2 = etal;     etal = eta; 
    dlta_tmp = sr2*vepln - cr2*dlta;
    veplnl   = cr2*vepln + sr2*dlta;
    dlta     = dlta_tmp;   eta = sr2*gama;   gama = -cr2*gama;
  end

%% Compute the current right plane rotation P{k-1,k}, P_12, P_23,...
  if iter > 1
    [cr1, sr1, gamal] = SymGivens2(gamal, dlta);
    vepln =   sr1*gama;
    gama  = - cr1*gama;
  end

%% Update xnorm  
  xnorml = xnorm;     ul4 = ul3;     ul3   = ul2;
  if iter > 2
    ul2 = (taul2 - etal2*ul4 - veplnl2*ul3) / gamal2;
  end
  if iter > 1
    ul = ( taul  - etal *ul3 - veplnl *ul2) / gamal;
  end
  xnorm_tmp = norm([xl2norm ul2 ul]);
  if abs(gama) > realmin && xnorm_tmp < maxxnorm
    u = (tau - eta*ul2 - vepln*ul) / gama; 
    if norm([xnorm_tmp u]) > maxxnorm
      u = 0;      flag = 6;
    end
  else
    u = 0;     flag = 9;
  end   
  xl2norm = norm([xl2norm ul2]);
  xnorm   = norm([xl2norm ul u]); 

%% Update w. Update x except if it will become too big 
  if (Acond < TranCond) && flag ~= flag0 && QLPiter==0  %% MINRES updates
    wl2 = wl;     wl = w;
    w   = (v - epln*wl2 - dlta_QLP*wl) * (1/gama_tmp);
    if xnorm < maxxnorm
      x = x + tau*w;    
    else
      flag = 6;
    end

  else %% MINRES-QLP updates
    QLPiter = QLPiter + 1;
    if QLPiter == 1
      xl2 = zeros(n,1);      
      if  (iter > 1) % construct w_{k-3}, w_{k-2}, w_{k-1}     
	if iter > 3
	  wl2 = gamal3*wl2 + veplnl2*wl + etal*w;
	end     % w_{k-3}
	if iter > 2
	  wl = gamal_QLP*wl + vepln_QLP*w;
	end     % w_{k-2}
	w = gama_QLP*w;     xl2 = x - wl*ul_QLP - w*u_QLP;
      end
    end
    if iter == 1
      wl2 = wl;      wl = v*sr1;     w  = -v*cr1;
    elseif iter == 2
      wl2 = wl; 
      wl  = w*cr1 + v*sr1;
      w   = w*sr1 - v*cr1;
    else
      wl2 = wl;      wl = w;         w  = wl2*sr2 - v*cr2;
      wl2 = wl2*cr2 + v*sr2;         v  = wl *cr1 + w*sr1;    
      w   = wl *sr1 - w*cr1;         wl = v;
    end
    xl2 = xl2 + wl2*ul2;
    x   = xl2 + wl *ul + w*u;    
  end

%% Compute the next right plane rotation P{k-1,k+1}
  gamal_tmp = gamal;
  [cr2,sr2,gamal] = SymGivens2(gamal,eplnn);  

%% Store quantities for transfering from MINRES to MINRES-QLP 
  gamal_QLP = gamal_tmp;     vepln_QLP = vepln;     gama_QLP = gama; 
  ul_QLP    = ul;            u_QLP     = u;

%% Estimate various norms    
  abs_gama = abs(gama);      Anorml = Anorm;
  Anorm = max([Anorm, pnorm, gamal, abs_gama]);                              
  if iter == 1
    gmin   = gama;    gminl = gmin;
  elseif iter > 1
    gminl2 = gminl;   gminl = gmin;    gmin = min([gminl2, gamal, abs_gama]);
  end
  Acondl   = Acond;     Acond   = Anorm/gmin;
  rnorml   = rnorm;     relresl = relres;     
  if flag ~= 9, rnorm = phi; end   
  relres   = rnorm / (Anorm*xnorm + beta1);
  rootl    = norm([gbar; dltan]);   
  Arnorml  = rnorml*rootl;
  relAresl = rootl / Anorm;            

%% See if any of the stopping criteria are satisfied.
  epsx = Anorm*xnorm*eps; 
  if (flag == flag0) || (flag == 9)
    t1 = 1 + relres;
    t2 = 1 + relAresl;
    if iter     >= maxit   , flag = 8; end  % Too many itns
    if Acond    >= Acondlim, flag = 7; end  % Huge Acond  
    if xnorm    >= maxxnorm, flag = 6; end  % xnorm exceeded its limit
    if epsx     >= beta1   , flag = 5; end  % x is an eigenvector
    if t2       <= 1       , flag = 4; end  % Accurate LS solution
    if t1       <= 1       , flag = 3; end  % Accurate Ax=b solution
    if relAresl <= rtol    , flag = 2; end  % Good enough LS solution
    if relres   <= rtol    , flag = 1; end  % Good enough Ax=b solution
  end

% The "disable" option allowed iterations to continue until xnorm
% became large and x was effectively a nullvector.
% We know that r will become a nullvector much sooner,
% so we now disable the disable option :)

%  if disable && (iter < maxit)
%    flag = 0;
%    if Axnorm < rtol*Anorm*xnorm
%      flag = 10;
%    end
%  end

  if flag == 2 || flag == 4 || flag == 6 || flag == 7   % Possibly singular
    iter  = iter - 1;
    Acond = Acondl;   rnorm = rnorml;   relres = relresl;
  else
    if ~isempty(resvec)
      resvec(iter+1) = rnorm;
      Aresvec(iter)  = Arnorml;
    end

    if show && mod(iter,lines) == 0
      if iter == 101
	lines = 10;     headlines = 20*lines;
      elseif iter == 1001
	lines = 100;    headlines = 20*lines;
      end
      if QLPiter == 1
	fprintf('%s', 'P')
      else
	fprintf('%s', ' ')
      end
      fprintf('%7g %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n',...
	  iter-1, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl, xnorml)
      if iter > 1 && mod(iter,headlines) == 1
	fprintf('\n%s\n', head)
      end
    end
  end
end % while


%% We have exited the main loop.
if QLPiter == 1
  fprintf('%s', 'P')
else
  fprintf('%s', ' ')
end
Miter = iter - QLPiter;

%% Compute final quantities directly.
r1      = b - minresxxxA(A,x) + shift*x;   % r1 is workspace for residual vector
rnorm   = norm(r1);
Arnorm  = norm(minresxxxA(A,r1) - shift*r1);
xnorm   = norm(x);
relres  = rnorm / (Anorm*xnorm + beta1);
relAres = 0;
if rnorm > realmin
  relAres = Arnorm / (Anorm*rnorm);
end

if ~isempty(Aresvec)
  Aresvec(iter+1) = Arnorm;
  Aresvec         = Aresvec(1:iter+1);
end
if ~isempty(resvec)
  resvec          = resvec(1:iter+1);
end

if show
  if rnorm > realmin
    fprintf('%7g %10.2e %10.2eD%10.2e %10.2eD%10.2e %10.2e %10.2e\n\n',...
            iter, rnorm, Arnorm, relres, relAres, Anorm, Acond, xnorm)
  else 
    fprintf('%7g %10.2e %10.2eD%10.2e %10.2e             %10.2e %10.2e\n\n',...
            iter, rnorm, Arnorm, relres,          Anorm, Acond, xnorm)
  end
end
fprintf('%s flag  =%7g  %s\n'                                , last, flag , msg(flag+2,:))
fprintf('%s iter  =%7g   (MINRES%7g, MINRES-QLP%7g)\n'       , last, iter , Miter, QLPiter)
fprintf('%s rnorm = %11.4e     rnorm  direct = %11.4e\n'     , last, rnorm, norm(r1))
fprintf('%s                         Arnorm direct = %11.4e\n', last, Arnorm)
fprintf('%s xnorm = %11.4e     xnorm  direct = %11.4e\n'     , last, xnorm, norm(x))
fprintf('%s Anorm = %11.4e     Acond         = %11.4e\n'     , last, Anorm, Acond)


%% Private functions
function p = minresxxxA( A, x )
  if isa(A,'function_handle')
    p = A(x);
  else
    p = A*x;
  end
  

function p = minresxxxM( M, x )
  persistent R 
  if isa(M,'function_handle')
    p = M(x);
  elseif ~exist('R','var') || isempty(R)
    R = chol(M);
  end 
  p = R'\x;
  p = R\p;

  
function [c,s,d] = SymGivens2(a,b)
  if b == 0,
    if a == 0, 
      c = 1; 
    else 
      c = sign(a);  % NOTE: sign(0) = 0 in MATLAB
    end 
    s = 0;
    d = abs(a);

  elseif a == 0
    c = 0;
    s = sign(b);
    d = abs(b);
  
  elseif abs(b) > abs(a)
    t = a/b;
    s = sign(b) / sqrt(1 + t^2); 
    c = s*t;
    d = b/s; % computationally better than d = a / c since |c| <= |s|

  else 
    t = b/a; 
    c = sign(a) / sqrt(1 + t^2); 
    s = c*t;
    d = a/c; % computationally better than d = b / s since |s| <= |c|
  end

%% end of minresQLP
