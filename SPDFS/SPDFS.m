%\min_{W'*W = I,||W||_{2,0} = k,\sum_{j=1}^c y_{ij}=1,y_{ij}\in [0,1],M}
%\sum_{i=1}^n\sum_{j=1}^c(||W'*x_i-m_j||_2^2)*y_{ij}^r/trace(W'*St*W)

function [opt_index,W] = SPDFS(X,c,phi,k,m)
% Input
% X: dim*num data matrix, each column is a data point
% c: number of classes
% phi: fuzzy exponent (> 1) for the partition matrix
% k: number of selected features
% m: reduced dimensionality

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix with k nonzero rows

[dim,n] = size(X);
NITER = 50;

%% ===================== Initialization =====================

%% initialize Y
Y = initfcm(c, n);
[Y,O] = step_fcm(X',Y,c,phi);

%% calculate satter matrix
St = X*X';
F = Y'.^phi;
Sm = X*(diag(F*ones(c,1))-F*diag((F'*ones(n,1)).^-1)*F')*X';

%% initialize W
% random initialization
W = orth(rand(dim,m)); 
% calculate lambda
lambda = trace(W'*Sm*W)/trace(W'*St*W);
% calculate A
S = Sm-lambda*St;
[~,eigvalue,~] = eig1(S,1);  
A = eigvalue*eye(dim)-S;
% initialize W with k nonzero rows
[~,W] = L20sparse(A,k,W,1);
C = O*W;

%% =======================  updating  =======================
%% outer loop
obj_o = [];
for iter_o = 1:NITER   
    
   %% update lambda
    if iter_o == 1
       lambda = trace(W'*Sm*W)/trace(W'*St*W);
    else
       lambda = P/Q;
    end
    
   %% calculate objective value
    obj_o = [obj_o;lambda];
    if iter_o>=2 && obj_o(iter_o-1)-obj_o(iter_o)<10^-4
       break;
    end
    
   %% inner loop
    obj_i = [];
    for iter_i = 1:NITER  
    
        % calculate objective value
        if iter_i == 1
           obj_i = [obj_i;0];
        else
           P = trace(W'*Sm*W);
           Q = trace(W'*St*W);
           obj_i = [obj_i;P-lambda*Q];
           if obj_i(iter_i-1)-obj_i(iter_i)<10^-4
              break;
           end 
        end

        % update Y
        data = W'*X;
        [Y,C] = step_fcm(data',Y,c,phi,C);
    
        % calculate satter matrix
        F = Y'.^phi;
        Sm = X*(diag(F*ones(c,1))-F*diag((F'*ones(n,1)).^-1)*F')*X';    
        
        % calculate A
        S = Sm-lambda*St;
        [~,eigvalue,~] = eig1(S,1);
        A = eigvalue*eye(dim)-S;
        
        % update W
        [opt_index,W] = L20sparse(A,k,W,NITER);
       
    end  
    
end

end

function [U_new, center] = step_fcm(data, U, cluster_n, expo, center_old)

if nargin < 5
   center_old = [];
end

mf = U.^expo;       
sumU = sum(mf,2);
ind = find(sumU==0);
center = mf*data./(sumU*ones(1,size(data,2)));
if ~isempty(ind)
   center(ind,:) = center_old(ind,:);  
end
dist = distfcm(center, data);       
tmp = dist.^(-2/(expo-1));     
U_new = tmp./(ones(cluster_n, 1)*sum(tmp));

end