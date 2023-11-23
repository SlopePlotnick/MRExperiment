[num_e3, txt_e3, raw_e3] = xlsread('data.xlsx', 'B2:B1001');
[num_e4, txt_e4, raw_e4] = xlsread('data_1e4.xlsx', 'B2:B10001');
[num_e5, txt_e5, raw_e5] = xlsread('data_1e5.xlsx', 'B2:B100001');

[f_e3, xi_e3, bw_e3] = ksdensity(num_e3);
[f_e4, xi_e4, bw_e4] = ksdensity(num_e4);
[f_e5, xi_e5, bw_e5] = ksdensity(num_e5);

disp('样本量：1000')
disp(bw_e3);
disp('样本量：10000')
disp(bw_e4);
disp('样本量：100000')
disp(bw_e5);