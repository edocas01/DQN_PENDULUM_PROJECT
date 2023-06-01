close all;
clear all;
maxvalue = 10;
dnu = 27;
DU = 2*maxvalue/dnu;

iu = (0:1:dnu);
for i = 1:dnu
    u(i) = generate_u(iu(i),dnu, DU);
    u_u(i) = generate_u_uniform(iu(i), dnu, DU);
end

plot(u, ones(dnu), '.r','LineWidth',2,'MarkerSize',25)
hold on
plot([-10,10],[1,1],'-k','LineWidth',0.08)
plot(u_u, ones(dnu)+0.2, '.b','LineWidth',2,'MarkerSize',25)
plot([-10,10],[1.2,1.2],'-k','LineWidth',0.08)
ylim([0.8,3])



function u = generate_u(iu, dnu, DU)
    if iu > dnu
        iu = dnu;
    elseif iu < 0
        iu = 0;
    end
    iu = iu - (dnu-1)/2;
    DUnew = 2*DU*abs(iu)/dnu;
    
    u = DUnew*iu;
end

function u = generate_u_uniform(iu, dnu, DU)
    if iu > dnu
        iu = dnu;
    elseif iu < 0
        iu = 0;
    end
    iu = iu - (dnu-1)/2; 
    u = DU*iu;
end
