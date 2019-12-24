figure;
x = linspace(-10,10,1000);
f = @(x) normpdf(x, 0, 1);
plot(x, f(x) ./ integral(f, -10, 10));
hold on;
f = @(x) normpdf(x, 0, 1).^2;
plot(x, f(x) ./ integral(f, -10, 10));

X = [];
for k = 1:100000
    s = normrnd(0,1, [100,1]);
    [i,j] = find(min(pdist(s)) == pdist2(s,s));
    X(end+1) = s(i(randi(2)));
end

histogram(X,'Normalization','pdf');
xlim([-10,10]);



figure;
x = linspace(0,20,1000);
f = @(x) gampdf(x, 2, 2);
plot(x, f(x) ./ integral(f, 0, 20));
hold on;
f = @(x) gampdf(x, 2, 2).^2;
plot(x, f(x) ./ integral(f, 0, 20));
    
X = [];
for k = 1:100000
    s = gamrnd(2,2, [100,1]);
    [i,j] = find(min(pdist(s)) == pdist2(s,s));
    %X(end+1) = s(i(randi(2)));
    X(end+1) = s(i(1));
    X(end+1) = s(i(2));
end

histogram(X,'Normalization','pdf');
xlim([0,10]);



figure;
x = linspace(0,1,1000);
f = @(x) betapdf(x, .8, .8);
plot(x, f(x) ./ integral(f, 0, 1));
hold on;
f = @(x) betapdf(x, .8, .8).^2;
plot(x, f(x) ./ integral(f, 0, 1));
    
X = [];
for k = 1:100000
    s = betarnd(.8,.8, [100,1]);
    [i,j] = find(min(pdist(s)) == pdist2(s,s));
    X(end+1) = s(i(randi(2)));
end

histogram(X,100,'Normalization','pdf');






%%%%%%%%%%%%%%%%%%%%

figure;
hold on;
x = linspace(-10,10,1000);

q = @(x) normpdf(x, 0, 1);
plot(x, q(x));

z = @(x) q(x).^2;
plot(x, z(x) ./ integral(z, -10, 10));

p = @(x) normpdf(x, 1, 1);
plot(x, p(x));

g = @(x) z(x) ./ p(x);
plot(x, g(x));

X = [];
for k = 1:1
    s = normrnd(0,1, [100,1]);
    [i,j] = find(min(pdist(s)) == pdist2(s,s));
    X(end+1) = s(i(randi(2)));
end
histogram(X,200,'Normalization','pdf');


