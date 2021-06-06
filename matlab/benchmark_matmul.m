

ns = [31, 44, 63, 89, 125, 177, 251, 354, 501, 707, 1000, 1412, 1995, 2818, 3981];
nruns = 5;

time_mean = zeros(1, length(ns));
time_std = zeros(1, length(ns));

for i = 1:length(ns)
    n = ns(i);
    disp(n);
    t = zeros(length(nruns), 1);
    for k = 1:nruns
        disp(k);
        [A, B, C] = build_matrices(n, n);
        tic
        X = A * B;
        t(k) = toc;
    end
    time_mean(i) = mean(t);
    time_std(i) = std(t);
end

loglog(ns, time_mean, '-o');
hold on;
% Plot standard deviation
y = [ns, fliplr(ns)];
in_between = [time_mean - time_std, fliplr(time_mean + time_std)];
h = fill(y, in_between, 'b');
set(h, 'facealpha', 0.4);

xlabel('n');
ylabel('time')

% Save data
dump_data = [ns', time_mean', time_std'];
csvwrite('benchmark_matmul.csv', dump_data)




