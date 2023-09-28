%% dataset setting
id=1;
d_name=['cc_1_d.txt'];
weight_name=['cc_1_w.txt'];
data_name=['cc_1_data.txt'];
P=10;

%% load dataset
data=load(data_name);
[capacity,d,weight]=generate_d_w(d_name,weight_name);
N=length(weight);
d=reshape(d,N,N);

%% data normalization
max_weight=max(weight);
weight=weight/max_weight;
capacity=capacity/max_weight;

%% parameters setting
POP=32;
M=1000;
T0=1.7;
ALPHA=5;
BETA=5;

[gbest,time,gbestx] = CNO_CC(d,weight,capacity,P,T0,ALPHA,BETA,M,POP);
%% It is also optional to use default parameters
%[gbest,time,gbestx] = CNO_CC(d,weight,capacity,P,T0,ALPHA,BETA);
%[gbest,time,gbestx] = CNO_CC(d,weight,capacity,P);

%% Save results
%pre = [cd,'/results/'];
filename = ['/example_',num2str(id),'_cno_cc.txt'];
savePath = [cd,filename];
writematrix([gbest,time,gbestx'],savePath,'Delimiter','\t','WriteMode','append')
disp(['Problem ',num2str(id),' is solved!'])


function [capacity,d,weight]=generate_d_w(d_name,weight_name)
d=load(d_name);
capacity_weight=load(weight_name);
capacity=capacity_weight(1,1);
weight=capacity_weight(2:end,1);
end