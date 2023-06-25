close all;
clear all;
maxvalue = 5;

vect_dnu = [3,17,35,51,91];

values = cell(length(vect_dnu),1);

figure(1)
for i = 1:length(vect_dnu)
    tmp = [];
    iu = (0:1:vect_dnu(i));
    for j = 1:vect_dnu(i)
        tmp(j) = generate_u(iu(j),vect_dnu(i),maxvalue);
    end
    ll = ['dnu_',num2str(vect_dnu(i))];
    plot(tmp,i*ones(vect_dnu(i),1),'.','LineWidth',2,'MarkerSize',25,'DisplayName',ll)
    hold on
    plot([-maxvalue,maxvalue],[i,i],'-k','LineWidth',0.08,'HandleVisibility','off')
end
xlim('padded')
ylim([-2,10])
legend('Location','east')


function u = generate_u(iu, dnu,maxvalue)
    if iu > dnu
        iu = dnu;
    elseif iu < 0
        iu = 0;
    end
    iu = iu - (dnu-1)/2;
    DU = 2*maxvalue/dnu;
    DUnew = 2*DU*abs(iu)/dnu;
    
    u = DUnew*iu;
end
