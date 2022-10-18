function [] = CNO_CC()
%% dataset setting
id=1;
d_name=['cc_1_d.txt'];
weight_name=['cc_1_w.txt'];
data_name=['cc_1_data.txt'];
N=100;
P=10;

%% load dataset
data=load(data_name);
[capacity,d,weight]=generate_d_w(d_name,weight_name);
d=reshape(d,N,N);
max_weight=max(weight);

%% data normalization
weight=weight/max_weight;
capacity=capacity/max_weight;

%% parameters setting
POP=32;
M=1000;
W=1;
C1=2;
C2=2;
T0=1.7;
ALPHA=5;
BETA=3;

%% multi-testing
test_times=20;
for times=0:test_times-1
    lbestx=zeros(N*P,POP);
    lbest=zeros(1,POP);
    best_flg=0;
    gbestx=zeros(N*P,1);
    gbest=0;
    m=0;
    gen=0;
    pre_m=50;
    lambda=zeros(P,POP);
    time=0;
    
    %% generate mutiple random feasible solution
    x=round(rand(N*P,POP));
    %[x] = generate_rand_feasible_solution(data,POP,P,N);
    initialx=x;
    initialv=unifrnd(-1,1,N*P,POP);
        
    s1=zeros(N,POP);
    s2=zeros(P,POP);
    t=zeros(N*P,POP);
    for pop=1:POP
        for i=1:N
            for p=1:P
                s1(i,pop)=s1(i,pop)+x((i-1)*P+p,pop);
                if x((i-1)*P+p,pop)==1
                    s2(p,pop)=s2(p,pop)+weight(i);
                    for ii=1:N
                        t((ii-1)*P+p,pop)=t((ii-1)*P+p,pop)+d(ii,i);
                    end
                end
            end
        end
    end
    for pop=1:POP
        total=0;
        const1=0;
        const2=0;
        for p=1:P
            term=0;
            element_num=0;
            for i=1:N
                if x((i-1)*P+p,pop)==1
                    term=term+t((i-1)*P+p,pop);
                    element_num=element_num+1;
                end
            end
            if element_num>1.5
                lambda(p,pop)=term/(2*element_num);
                total=total+lambda(p,pop);
            else
                lambda(p,pop)=0;
            end
        end
        for i=1:N
            const1=const1+10000*(s1(i,pop)-1)*(s1(i,pop)-1);
        end
        for p=1:P
            const2_temp=s2(p,pop)-capacity;
            if const2_temp>0
                const2=const2+0.5;
            end
        end
        total=total+const1+const2;
        lbest(1,pop)=total;
        lbestx(:,pop)=x(:,pop);
    end

    tic
    while m<M
        s1=zeros(N,POP);
        s2=zeros(P,POP);
        t=zeros(N*P,POP);
        for pop=1:POP
            for i=1:N
                for p=1:P
                    s1(i,pop)=s1(i,pop)+initialx((i-1)*P+p,pop);
                    if initialx((i-1)*P+p,pop)==1
                        s2(p,pop)=s2(p,pop)+weight(i);
                        for ii=1:N
                            t((ii-1)*P+p,pop)=t((ii-1)*P+p,pop)+d(ii,i);
                        end
                    end
                end
            end
        end
        
        [s1,s2,lbestx,lbest,lambda,best_flg]=fully_parallel_update_GPU(P,N,POP,initialx,s1,s2,t,T0,ALPHA,BETA,lambda,d,weight,capacity,best_flg,lbestx,lbest);
        s1_num=0;
        for pop=1:POP
            if isempty(find(s1(:,pop)~=1))
                s1_num=s1_num+1;
            end
        end
        disp(s1_num)
        
        s2_num=POP;
        for pop=1:POP
            for p=1:P
                if s2(p,pop)>capacity
                    s2_num=s2_num-1;
                    break
                end
            end
        end
        disp(s2_num)
        
        [gbest,gbestx,m,pre_m,time]=generate_global_best(gbest,gbestx,lbest,lbestx,gen,m,pre_m,time);
        
        if m-pre_m>0
            pre_m=pre_m+50;
            unconverage_num=0;
            for pop=1:POP
                if (lbest(1,pop)-gbest)>0.01
                    unconverage_num=unconverage_num+1;
                end
            end
            if (unconverage_num<3)
                initialv=unifrnd(-1,1,N*P,POP);
                lbestx=zeros(N*P,POP);
                lbest=zeros(1,POP);
                best_flg=0;
            end
        else
            initialv=W*initialv+C1*rand(1,POP).*(lbestx-initialx)+C2*rand(1,POP).*(gbestx-initialx);
        end
        probability=1./(1.+exp(-initialv));
        zero_set=probability<rand(N*P,POP);
        initialx(~zero_set)=1;
        initialx(zero_set)=0;
        
        gen=gen+1;
    end
    pre = [cd,'/results/'];
    filename = ['example_',num2str(id),'_cno_cc.txt'];
    savePath = [pre,filename];
    writematrix([gbest,time,gbestx'],savePath,'Delimiter','\t','WriteMode','append')
    disp(['Problem ',num2str(id),' is solved!'])
end
end

function [gbest,gbestx,m,pre_m,time]=generate_global_best(gbest,gbestx,lbest,lbestx,it,m,pre_m,time)
gb=min(lbest);
id=find(lbest==gb);

if (it==0||gb<gbest)
    gbest=gb;
    gbestx=lbestx(:,id(1,1));
    m=0;
    pre_m=50;
    time=toc;
else
    m=m+1;
end
fprintf('%d = %f\n',it,gbest)
end

function [s1,s2,lbestx,lbest,lambda,gen]=fully_parallel_update_GPU(P,N,POP,initialx,s1,s2,t,T0,ALPHA,BETA,lambda,d,weight,capacity,gen,lbestx,lbest)
x=initialx; 
u=zeros(N*P,POP);
%changed_num=0;

for it=0:100
    T=T0*0.2^it;
    for pop=1:POP
        for i=1:N
            for p=1:P
                x_pos=x((i-1)*P+p,pop);
                s1_i=s1(i,pop);
                s2_i=s2(p,pop);
                P2=0;
                if x_pos==1
                    if s2_i>capacity
                        P2=0.5;  % 'if' is slow, use this instead of the following code.
                        %                         if s2_i-weight(i+1,1)>capacity
                        %                             P2=weight(i+1,1);
                        %                         else
                        %                             P2=s2_i-capacity;
                        %                         end
                    end
                else
                    if s2_i+weight(i,1)>capacity
                        P2=0.5;  % 'if' is slow, use this instead of the following code.
                        %                         if s2_i>capacity
                        %                             P2=weight(i+1,1);
                        %                         else
                        %                             P2=s2_i+weight(i+1,1)-capacity;
                        %                         end
                    end
                end
                dedx=t((i-1)*P+p,pop)+ALPHA*(s1_i-0.5-x_pos)+BETA*P2-lambda(p,pop);
                u((i-1)*P+p,pop)=u((i-1)*P+p,pop)+dedx;
            end
        end
    end
    for pop=1:POP
        for i=1:N
            for p=1:P
                x_pos=x((i-1)*P+p,pop);
                Probability=1/(1+exp(u((i-1)*P+p,pop)/T));
                rand_num=rand;
                if rand_num<Probability
                    x((i-1)*P+p,pop)=1;
                elseif rand_num>Probability
                    x((i-1)*P+p,pop)=0;
                end
                
                if x((i-1)*P+p,pop)~=x_pos
                    %changed_num=changed_num+1;
                    deltx=x((i-1)*P+p,pop)-x_pos; %看是变大了还是变小了
                    s1(i,pop)=s1(i,pop)+deltx;
                    s2(p,pop)=s2(p,pop)+weight(i,1)*deltx;
                    for j=1:N
                        t((j-1)*P+p,pop)=t((j-1)*P+p,pop)+d(i,j)*deltx;
                    end
                end
            end
        end
    end
end
% disp(changed_num)
%disp(it)

for pop=1:POP
    total=0;
    const1=0;
    const2=0;
    
    for p=1:P
        term=0;
        element_num=0;
        for i=1:N
            if x((i-1)*P+p,pop)==1
                term=term+t((i-1)*P+p,pop);
                element_num=element_num+1;
            end
        end
        if element_num>1.5
            lambda(p,pop)=term/(2*element_num);
            total=total+lambda(p,pop);
        else
            lambda(p,pop)=0;
        end
    end
    
    for i=1:N
        const1=const1+10000*(s1(i,pop)-1)*(s1(i,pop)-1);
    end
    
    for p=1:P
        const2_temp=s2(p,pop)-capacity;
        if const2_temp>0
            const2=const2+0.5;
        end
    end
    total=total+const1+const2;
    
    if total<lbest(1,pop)
        lbest(1,pop)=total;
        lbestx(:,pop)=x(:,pop);
    end
end
end

function [x] = generate_rand_feasible_solution(data,POP,P,N)
x=zeros(N*P,POP);
for pop=1:POP
    result=kmeans(data,P);
    for i=0:N-1
        x(i*P+result(i+1,1),pop)=1;
    end
end
end

function [capacity,d,weight]=generate_d_w(d_name,weight_name)
d=load(d_name);
capacity_weight=load(weight_name);
capacity=capacity_weight(1,1);
weight=capacity_weight(2:end,1);
end