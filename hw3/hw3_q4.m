x1n = [1,4,5,3,4,8];
x2n = [10,4,6,16,2,8];

x1p = [8,7,10,4];
x2p = [7,7,14,10];

plot(x1n,x2n,'*','markers',10);
axis([0,17,0,17]);
xlabel('x1');
ylabel('x2');
hold on
plot(x1p,x2p,'+','markers',10);
hold off

