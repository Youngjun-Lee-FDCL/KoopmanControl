clear; close all; clc;
set(0, 'DefaultLineLineWidth', 1.5);
set(0,'DefaultFigureColor','white');
set(0,'defaultAxesFontSize',15);

prefix = "lqr";
data = readtable(prefix+"1.csv");
figure(1);
plot(data.t, data.cmd,"--k"); hold on; grid on; xlabel("Time (sec)"); ylabel("Angle of attack (deg)");
figure(4);
plot(data.t, 30.0*ones(length(data.t),1),"--r"); hold on;
plot(data.t, -30.0*ones(length(data.t),1),"--r"); grid on; xlabel("Time (sec)"); ylabel("Fin deflection angle (deg)")
ylim([-32,32]);
color = ColorBand(20);
for i = 1:20
    data = readtable(prefix+num2str(i)+".csv");
    figure(1);
    plot(data.t, data.alpha, Color=color(i, :)); 
    figure(2); 
    plot(data.t, data.mach,Color=color(i, :)); hold on;
    figure(3);
    plot(data.t, data.h,Color=color(i, :)); hold on;
    figure(4);
    plot(data.t, data.u,Color=color(i, :));
end
figure(1);
legend("Cmd"); ylim([-20,20])