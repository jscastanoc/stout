
clear; close all; clc
addpath('utils/')

%% Load data
load('data/head_model.mat')
load('data/VC4001montreal.mat')

rng(100)

%% Setup spatial dictionary
distmat = squareform(distmat);
B = exp(-(distmat).^2);
B = nip_blobnorm(B,'norm',2);
for i = 1 :size(B,2)
    idx_ins = find(B(i,:) < max(B(i,:))*0.001);
    B(i,idx_ins) = 0;
end

%% Simulate neural data
t = 0:1/100:1;
act = [sin(2*pi*8*t)]; 
sim_pos = [10 0 10];
[J, idx_act] = simulate_activity(head_model.cortex.vc,sim_pos, act, randn(1,3), t);
index = (1:3:size(head_model.L,2));
for i = 0:2
    J(index+i,:) = B*J(index+i,:); 
end

%% Calculate pseudo eeg
data.x = head_model.L*J;

%% STOUT
[J_rec,~] = stout(data.x,head_model.L,B,'tstep',4,'wsize',80,'sreg',90,'treg',30,'tol',1e-1,'optimres',false);

%% Calculate activity for a given time instant
J3d = sqrt(sum(J_rec.^2,2));


%% Visualize the results
load('data/cm8.mat');

options3d.view= [90 0];
options3d.colormap = cm8;
options3d.axes = gca;
figure('Units','normalized','position',[0.2 0.2 0.3 0.3]);
reconstruction3d(head_model.cortex, J3d , options3d);hold on
scatter3(head_model.cortex.vc(idx_act,1),...
    head_model.cortex.vc(idx_act,2),....
    head_model.cortex.vc(idx_act,3),'fill')

figure('Units','normalized','position',[0.2 0.2 0.3 0.3]);
plot(t,J_rec)
