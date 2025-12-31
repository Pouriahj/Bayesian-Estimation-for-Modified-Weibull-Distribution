function Y = mod_wblpdf(x, V, s0, B, m)
% modified weibull PDF
% Inputs:
% I) x is the variable, 
% II) V is the volume and length in some cases
% III) "s0", "B" and "m" are the parameters of the PDF
Y = (m./(s0.^m)).*((V.^B).*(x.^(m-1)).*exp(-(V.^B).*((x./s0).^m)));
end