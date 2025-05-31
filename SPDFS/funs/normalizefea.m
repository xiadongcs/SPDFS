%% Preprocessing
function [newfea] = normalizefea(num, oldfea)
X0 = oldfea;
mX0 = mean(X0);
X0 = X0 - ones(num,1)*mX0;
scal = 1./sqrt(sum(X0.*X0)+eps);
scalMat = sparse(diag(scal));
newfea = X0*scalMat;
end