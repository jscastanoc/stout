% Demo script to localize visual/auditory ERPs using STOUT

clear; close all; clc

condition = 'visual';
sub = 10;
addpath('utils/')
load('data/head_model.mat')
load('data/VC4001montreal.mat')
distmat = squareform(distmat);
load(['data/' condition '/s' num2str(sub) 'nt.mat'])

%% Prepare variables
B = exp(-(distmat).^2);
B = blobnorm(B,'norm',2);
for i = 1 :size(B,2)
    idx_ins = find(B(i,:) < max(B(i,:))*0.001);
    B(i,idx_ins) = 0;
end
data.x = data.x(:,find(ismember(data.clab,head_model.clab)))';
head_model.L = head_model.L(find(ismember(head_model.clab,data.clab)),:);

% average reference
transM = eye(size(head_model.L,1))-(1/size(head_model.L,1))*ones(size(head_model.L,1));
data.x = transM*data.x;
head_model.L = transM*head_model.L;

% depth compensation
[head_model.L, extras] = depthcomp(head_model.L,...
    struct('type','Lnorm','gamma',0.3));
Winv = extras.Winv;

%% Compute reconstruction
[J_rec,~] = stout(data.x,head_model.L,B,'tstep',4,'wsize',80,...
    'Winv',Winv,'sreg',100,'treg',1,'tol',1e-1,'optimres',true);

%% Visualize Results
[~,t0] = sort(abs(data.t-200));
J3d = sqrt(sum(J_rec(:,t0(1)).^2,2));
load('data/cm8.mat');
options3d.view= [-90 0];
options3d.colormap = cm8;
options3d.axes = gca;
figure('Units','normalized','position',[0.2 0.2 0.3 0.3]);
reconstruction3d(head_model.cortex, J3d , options3d);

figure('Units','normalized','position',[0.2 0.2 0.3 0.3]);
plot(J_rec')
