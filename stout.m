function [J_rec extras] = stout(y,L,B,varargin)
% function [J_rec extras] = stout(y,L,varargin)
%  Input:
%         y -> NcxNt. Matrix containing the data,
%         L -> Ncx3Nd. Lead Field matrix
%         Additional options -> Key - Value pair:
%                 'sreg' -> scalar. Percentage of spatial
%                       regularization (between 0 and 1).
%                 'treg' -> scalar. Percentage of temporal
%                       regularization (between 0 and 1).
%                 a -> scalar. Time shift for the time frequency
%                       transform.
%                 m -> scalar. Frequency bins for the time frequency
%                       transform.
%                 tol -> Scalar. stopping criterion
%   Output:
%         J_rec -> 3NdxNt. Reconstructed activity (solution)
%
% Sebastian Castano-Candamil
% sebastian.castano@blbt.uni-freiburg.de
% Apr. 2015
%
% DISCLAIMER:
% The implementation of the FISTA solver is literal port of the corresponding
% python implementation found in mne-python (including tf-transform)
% https://github.com/mne-tools/mne-python
%
Ndor = size(L,2);
idx = 1:1:Ndor;
L_or = L;

[Nc Nd] = size(L);
Nt = size(y,2);

p = inputParser;

def_a = 4;
def_m = 80;
def_sreg = 90;
def_treg= 30;
def_maxiter = 500;
def_tol = 1e-3;
def_lipschitz = [];
def_optimres = true;
def_Winv = [];

addParamValue(p,'tstep',def_a);
addParamValue(p,'wsize',def_m);
addParamValue(p,'sreg',def_sreg);
addParamValue(p,'treg',def_treg);
addParamValue(p,'maxiter',def_maxiter);
addParamValue(p,'tol',def_tol);
addParamValue(p,'lipschitz',def_lipschitz);
addParamValue(p,'optimres',def_optimres);
addParamValue(p,'Winv',def_Winv)

parse(p,varargin{:})
options = p.Results;

tol = options.tol;
sreg = options.sreg;
treg = options.treg;
a = options.tstep;
M = options.wsize;
lipschitz_k = options.lipschitz;
maxiter = options.maxiter;
avg_res = optim_res_fct(y, L,options.Winv);

Ltemp = translf(L);
clear L
for i = 1:3
    L(:,:,i) = Ltemp(:,:,i)*B;
end
L = translf(L);
c = stft(y,M,a);
T = size(c,3);
K = size(c,2);
Z = sparse(0,K*T);
Y = sparse(Nd,K*T);
J_rec = sparse(Nd,Nt);

tau = 1;
tempGY = L'*y;
aux = sum(reshape(tempGY.^2',[],Nd/3)',2);
basepar = 0.01*sqrt(max(aux(:)));
L = L/basepar;
clear tempGY;

R = y;
active_set = logical(sparse(1,Nd));
Y_time_as = [];
Y_as = [];
if isempty(options.lipschitz)
    lipschitz_k = 1.1*lipschitz_contant(y, L, 1e-3, a, M);
end
fprintf('Lipschitz constant : %s \n', lipschitz_k)
mu_lc = sreg/lipschitz_k;
lambda_lc = treg/lipschitz_k;
stop =false;
fprintf('Running FISTA algorithm... \n');
eta = 0;
rescum = [];
error = inf;
res_0 = inf;
nn = 1;
while true
    rev_line = '';
    Z = sparse(0,K*T);
    Y = sparse(Nd,K*T);
    J_rec = sparse(Nd,Nt);
    tau = 1;
    active_set = logical(sparse(1,Nd));
    Y_time_as = [];
    Y_as = [];
    for i = 1:maxiter
        tic;
        Z_0 = Z;
        active_set0 = active_set;
        if ~isempty(Y_time_as) && sum(active_set)<Nc
            GTR = L'*R/lipschitz_k;
            A = GTR;
            A(Y_as,:) = A(Y_as,:) + Y_time_as;
            [~, active_set_l21] = prox_l21(A,mu_lc,3);
                temp = stft(GTR(find(active_set_l21),:),M,a);
                temp = reshape(temp,sum(active_set_l21),[]);
                Baux = Y(find(active_set_l21),:)+temp;
                [Z, active_set_l1] = prox_l1(Baux,lambda_lc,3);
                active_set_l21(find(active_set_l21))= active_set_l1;
                active_set_l1 = active_set_l21;            
        else
            temp = stft(R,M,a);
            temp = reshape(temp,Nc,[]);
            Y = Y + L'*temp/lipschitz_k;
            [Z, active_set_l1] = prox_l1(Y,lambda_lc,3);
        end
        [Z, active_set_l21] = prox_l21(Z,mu_lc,3);
        active_set = active_set_l1;
        active_set(find(active_set_l1)) = active_set_l21;
        error_0 = error;
        if norm(active_set - active_set0) == 0
            error = norm(abs(Z-Z_0))/norm(abs(Z_0));
        else
            error = inf;
        end
        msg = sprintf('Iteration # %d, Stop: %d, Time per iter: %f \nDipoles~=0: %d \nRegPar S:%d T:%d \n'...
            ,i,error,eta,sum(full(active_set))/3, mu_lc*lipschitz_k, lambda_lc*lipschitz_k);
        fprintf([rev_line, msg]);
        rev_line = repmat(sprintf('\b'),1,length(msg));
        if i < options.maxiter
            tau_0 = tau;
            tau = 0.5+sqrt(1+4*tau_0^2)/2;
            Y = sparse(Nd,K*T);
            dt = (tau_0-1)/tau;
            Y(find(active_set),:) = (1 + dt)*Z;
            Y(find(active_set0),:)= Y(find(active_set0),:) - dt*Z_0;
            Y_as = active_set0 | active_set;
            
            act_Y_as = find(Y_as);
            
            temp = reshape(full(Y(act_Y_as,:)), length(act_Y_as),K,T);
            Y_time_as = istft(temp,a,Nt);
            if isempty(Y_time_as)
                R = y;
            else
                R = y - L(:, act_Y_as)*Y_time_as(:,1:Nt);
            end
        else
            disp('')
        end
        stop = error < tol || ( sum(full(active_set))==0 && i > 1);
        if stop
            disp('Converged')
            break
        end
        eta = toc;
    end
    fprintf(' Done!... \nTransforming solution to the time domain: \n%d non-zero time series \n'...
        , sum(full(active_set)))
    temp = reshape(full(Z),[],K,T);
    temp = istft(temp,a,Nt);
    
    if ~isempty(temp)
        J_recf = sparse(Nd,size(temp,2));
        J_recf(find(active_set),:) = temp;
        J_recf = J_recf(:,1:Nt);
        Jf = zeros(Ndor,Nt);
        Jf(idx,:) = J_recf;
        Jf = Jf*norm(y,2)/norm(L_or*Jf,2);
        term_rec = L_or*Jf;
    else
        term_rec = NaN;
        Jf = [];
    end
    term_or = y;
    resnorm = norm(term_rec-term_or, 'fro');
    fprintf('RESNORM = %8.3e MAX:%1.3e\n',resnorm,avg_res);
    Jf_0 = Jf;
    J_rec = Jf;
    if (nn >= options.maxiter) || ((resnorm < avg_res) && (options.optimres)) || ((sum(active_set) > 0) && ( ~options.optimres))
        break;
    else
        mu_lc = 0.6*mu_lc;
        lambda_lc = 0.6*lambda_lc;
    end
    nn = nn+1;
end
J_rect = translf(J_rec');
J_rect = permute(J_rect,[2 1 3]);
for i = 1:3
    J_r(:,:,i) = B*J_rect(:,:,i);
end
J_rec = permute(J_r,[2 1 3]);
J_rec = translf(J_rec)';
if ~isempty(options.Winv)
    J_est = translf(J_rec');
    siJ = size(J_est);
    J_rec = permute(reshape(full(reshape(permute(J_est, [1 3 2]), siJ(1), [])*options.Winv), siJ(1), siJ(3), siJ(2)), [1 3 2]);
    J_rec = translf(J_rec)';
end
extras.active_set = active_set;
end

function [Y active_set] = prox_l21(Y,mu,n_orient)
n_pos = size(Y,1)/n_orient;
rows_norm = sqrt(sum(reshape(abs(Y).^2',[],n_pos)',2));
shrink = max(1 - mu./max(rows_norm,mu),0);
active_set = (shrink > 0);
shrink = shrink(active_set);
if n_orient>1
    active_set = repmat(active_set,1,n_orient);
    active_set = reshape(active_set',n_orient,[]);
    active_set = active_set(:)';
end
temp = reshape(repmat(shrink,1,n_orient),length(shrink),n_orient)';
temp = temp(:);
Y = Y(find(active_set),:).*repmat(temp,1,size(Y,2));
end

function [Y active_set] = prox_l1(Y,lambda,n_orient)
n_pos = size(Y,1)/n_orient;
norms = sqrt(sum(reshape((abs(Y).^2),n_orient,[]),1));
shrink = max(1-lambda./max(norms,lambda),0);
shrink = reshape(shrink',n_pos,[]);
active_set = logical(sum(shrink,2));
shrink = shrink(find(active_set),:);
active_set = repmat(active_set,1,n_orient);
active_set = reshape(active_set',n_orient,[]);
active_set = active_set(:)';
Y = Y(find(active_set),:);
if length(Y) > 0
    for i = 1:n_orient
        Y(i:n_orient:size(Y,1),:) = Y(i:n_orient:size(Y,1),:).*shrink;
    end
end
end

function k = lipschitz_contant(y, L, tol, a, M)
Nt = size(y,2);
Nd = size(L,2);
iv = ones(Nd,Nt);
v = stft(iv,M,a);
T = size(v,3);
K = size(v,2);
l = 1e100;
l_old = 0;
fprintf('Lipschitz constant estimation: \n')
rev_line = '';
xx = tic;
for i = 1 : 100
    tic
    msg = sprintf('Iteration = %d\nDiff:%d\nLipschitz Constant: %d\nTime per iteration %d\n',i,abs(l-l_old)/l_old,l,toc);
    fprintf([rev_line, msg]);
    rev_line = repmat(sprintf('\b'),1,length(msg));
    l_old = l;
    aux = istft(v,a,Nt);
    iv = real(aux);
    Lv = L*iv;
    LtLv = L'*Lv;
    w = stft(LtLv, M,a);
    l = max(max(max(abs(w))));
    v = w/l;
    if abs(l-l_old)/l_old < tol
        break
    end
end
fprintf('\n');
k = l;
toc(xx)
end

function avg_res = optim_res_fct(y, L,Winv)
Nd = size(L,2);
Q = speye(Nd);
inv_Lap = Q;
avg_res = 0;
for i = 1:10
    rand_idx = randperm(size(L,1));
    idx_tr = rand_idx(1:round(size(L,1)*0.5));
    idx_te = rand_idx(round(size(L,1)*0.5)+1:end);
    eye_Nc = speye(length(idx_tr));
    iLAP_LT = inv_Lap*L(idx_tr,:)';
    gcv_fun = @(alpha) gcv(y(idx_tr,:),L(idx_tr,:),alpha, inv_Lap, iLAP_LT, eye_Nc);
    optionsopt = optimset('tolX',1e-6);
    alpha = fminsearch(gcv_fun, 1,optionsopt);
    invT = iLAP_LT/(L(idx_tr,:)*iLAP_LT+abs(alpha)*eye_Nc);
    J_rec = invT*y(idx_tr,:);
    if ~isempty(Winv)
        J_est = translf(J_rec');
        siJ = size(J_est);
        J_rec = permute(reshape(full(reshape(permute(J_est, [1 3 2]), siJ(1), [])*Winv), siJ(1), siJ(3), siJ(2)), [1 3 2]);
        J_rec = translf(J_rec)';
    end
    J_rec = J_rec*norm(y(idx_te,:),2)/norm(L(idx_te,:)*J_rec,2);
    avg_res = avg_res + norm(y(idx_te,:)-L(idx_te,:)*J_rec,'fro')/10;
end
    avg_res = avg_res*(1/0.75); % Relax convergence criteria (require only 75% of variance explained by MNE)
end

function gcv_val = gcv(y,L,alpha, inv_Lap, iLAP_LT, eye_Nc)
T = iLAP_LT/(L*iLAP_LT+abs(alpha)*eye_Nc);
x_est = T*y;
A = norm(L*x_est - y,2);
gcv_val = sum(diag(A*A'))/trace((eye_Nc-L*T))^2;
end

function [ X] = stft(x, wsize, tstep)
if isempty(x)
    X = [];
    return
end

[Nc, Nt] = size(x);
n_step = ceil(Nt/tstep);
n_freq = wsize/2 + 1;
X = zeros(Nc,n_freq, n_step);

twin = [0.5:wsize+0.5-1];
win = sin(twin/wsize*pi);
win2 = win.^2;

swin = zeros(1,(n_step - 1)*tstep + wsize);
for t = 1:n_step
    swin((t-1)*tstep+1:(t-1)*tstep + wsize) = swin((t-1)*tstep+1:(t-1)*tstep + wsize) + win2;
end
swin = sqrt(wsize*swin);

xp = zeros([Nc, wsize + (n_step-1)*tstep]);
xp(:,(wsize-tstep)/2+1:(wsize-tstep)/2 + Nt) = x;
x = xp;

for t = 1:n_step
    wwin = win./swin((t-1)*tstep+1:(t-1)*tstep+wsize);
    wwin = repmat(wwin,Nc,1);
    frame = x(:, (t-1)*tstep+1:(t-1)*tstep+wsize).*wwin;
    fframe = fft(frame,[],2);
    X(:,:,t) = fframe(:,1:n_freq);
end


end


function [x] = istft(X, tstep, Tx)

if isempty(X)
    x = [];
    return
end
[Nc, n_win, n_step] = size(X);

wsize = 2*(n_win-1);
Nt = n_step * tstep;
x = zeros([Nc, Nt+wsize-tstep]);

twin = [0.5:wsize+0.5-1];
win = sin(twin/wsize*pi);
win2 = win.^2;

swin = zeros(1,(n_step - 1)*tstep + wsize);
for t = 1:n_step
    swin((t-1)*tstep+1:(t-1)*tstep + wsize) = swin((t-1)*tstep+1:(t-1)*tstep + wsize) + win2;
end
swin = sqrt(swin/wsize);
fframe = zeros([Nc, n_win+wsize/2-1]);
for t = 1:n_step
    fframe(:,1:n_win) = X(:,:,t);
    fframe(:,n_win+1:end) = conj(X(:,(wsize/2):-1:2,t));
    frame = ifft(fframe,[],2,'symmetric');
    wwin = win ./ swin((t-1)*tstep+1:(t-1)*tstep+wsize);
    wwin = repmat(wwin,Nc,1);
    x(:,(t-1)*tstep+1:(t-1)*tstep+wsize) = x(:,(t-1)*tstep+1:(t-1)*tstep+wsize) + real(conj(frame).*wwin);
end
x = x(:,(wsize-tstep)/2:(wsize-tstep)/2+Nt+1);
x = x(:,1:Tx);

end

