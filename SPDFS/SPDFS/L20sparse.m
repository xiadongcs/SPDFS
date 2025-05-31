function [opt_index,W] = L20sparse(A,k,W0,NITER)
% Input
% A: positive semi-definite matrix
% k: number of selected features
% W0: initialized W
% NITER: number of iterations

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix with k nonzero rows

%% ===================== Initialization =====================

%% initialize W
[dim,m] = size(W0);
W = W0;

%% =======================  updating  =======================
obj = [];
error = [];
Wr = cell(NITER);
for iter = 1:NITER
    
   %% calculate objective value
    obj = [obj; trace(W'*A*W)];
%     if iter>=2 && obj(iter)-obj(iter-1)<10^-4       % convergence in objective
%        break;
%     end

   %% convergence condition
    Wr{iter} = W;
    if iter>=2
       diff = norm(Wr{iter}-Wr{iter-1},'fro')^2;
       error = [error; diff]; 
       if diff == 0                                   % convergence to a fixed W
          break;
       end
    end
  
   %% update W
    P = A*W*pinv(W'*A*W)*W'*A;
    [~, ind] = sort(diag(P), 'descend');
    opt_index = sort(ind(1:k));
    Aopt = A(opt_index, opt_index);
    V = eig1(Aopt, m);
    W = zeros(dim,m);
    W(opt_index, :) = V;
   
end

end