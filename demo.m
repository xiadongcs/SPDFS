clc;
close all;

addpath('data'); addpath('funs'); addpath('SPDFS');
Files = dir(fullfile('data', '*.mat'));
Max_datanum = length(Files);
 
for data_num = 1:Max_datanum   
    Dname = Files(data_num).name;
    disp(['***********The test data name is: ***' num2str(data_num) '***'  Dname '****************'])
    load(Dname);
    
   %% create folder
    file_path = 'Results/SPDFS/';    
    folder_name = Dname(1:end-4);  
    file_path_name = strcat(file_path,folder_name);
    if exist(file_path_name,'dir') == 0   
       mkdir(file_path_name);
    end
    file_mat_path = [file_path_name '/'];
    
   %% preprocessing
    fea = X; gnd = Y; c = length(unique(Y));
    num = size(fea,1);
    fea = normalizefea(num, fea); 

   %% feature selection
    k = 50:50:300; phi_value = 1.1:0.1:1.3; rep = 20;
    for k_i = 1:length(k) 
        feanum = k(k_i); 
        m_inf = floor(feanum/3);
        m = m_inf:2:feanum; 
        result_mean = cell(length(phi_value),length(m));
        result_std = cell(length(phi_value),length(m));
        for phi_i = 1:length(phi_value)
            phi = phi_value(phi_i);
            for m_i = 1:length(m)
                dim = m(m_i);
                        
                [feature_id,W] = SPDFS(fea',c,phi,feanum,dim);
                X_new = fea(:,feature_id);
             
                results = zeros(rep,7);
                for rep_i = 1:rep       
                    lab = litekmeans(X_new,c,'Replicates',1);
                    results(rep_i,:) = ClusteringMeasure(gnd,lab); 
                end
                result_mean(phi_i,m_i) = {mean(results,1)};
                result_std(phi_i,m_i) = {std(results,0,1)};
 
                file_name = ['result_feanum_' num2str(feanum) '.mat'];
                save([file_mat_path,file_name],'result_mean','result_std');  
                         
            end
        end    
    end           
end