function [N,P,d,weight,capacity,data] = load_data2(id)
if id==1
    weight_name=['cc_1_w.txt'];
    data_name=['cc_1_data.txt'];
    N=100;
    P=10;
elseif id==2
    weight_name=['cc_2_w.txt'];
    data_name=['cc_2_data.txt'];
    N=200;
    P=15;
elseif id==3
    weight_name=['cc_3_w.txt'];
    data_name=['cc_3_data.txt'];
    N=300;
    P=25;
elseif id==4
    weight_name=['cc_4_w.txt'];
    data_name=['cc_4_data.txt'];
    N=300;
    P=20;
elseif id==5
    weight_name=['cc_5_w.txt'];
    data_name=['cc_5_data.txt'];
    N=402;
    P=30;
elseif id==6
    weight_name=['cc_6_w.txt'];
    data_name=['cc_6_data.txt'];
    N=402;
    P=40;
elseif id==7
    weight_name=['cc_7_w.txt'];
    data_name=['cc_7_data.txt'];
    N=318;
    P=15;
elseif id==8
    weight_name=['cc_8_w.txt'];
    data_name=['cc_8_data.txt'];
    N=1000;
    P=6;
elseif id==9
    weight_name=['cc_9_w.txt'];
    data_name=['cc_9_data.txt'];
    N=724;
    P=10;
elseif id==10
    weight_name=['cc_10_w.txt'];
    data_name=['cc_10_data.txt'];
    N=2000;
    P=6;
elseif id==11
    weight_name=['cc_11_w.txt'];
    data_name=['cc_11_data.txt'];
    N=318;
    P=40;
elseif id==12
    weight_name=['cc_12_w.txt'];
    data_name=['cc_12_data.txt'];
    N=1304;
    P=10;
end

data=load(data_name);
d=squareform(pdist(data));
capacity_weight=load(weight_name);
capacity=capacity_weight(1,1);
weight=capacity_weight(2:end,1);

max_weight=max(weight);
weight=weight/max_weight;
capacity=capacity/max_weight;

end

