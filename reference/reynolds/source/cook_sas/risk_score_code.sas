* SAS code for computing AHA/ACC, ATP III, and Reynolds risk scores;
* Created by Nancy Cook (BWH, HMS, HSPH) 08/25/2014 ;

* Variables:
	sex = 1 if male, 2 if female
	black = 1 if black, 0 if non-black
	lnage = log(age)
	lnchol = log(total cholesterol)
	lnhdl = log(HDL cholesterol)
	lnsbp = log(SBP)
	currsmoke = current smoker

	For AHA/ACC only:
	lsbp_trt = log (SBP) if treated, 0 if untreated
	lsbp_untrt = log (SBP) if NOT treated, 0 if treated
	diabetes = 1 if diabetic, 0 if non-diabetic

	For ATP-III only:
	trtsbp = 1 if BP treated, 0 if not treated

	For Reynolds only:
	lncrp = log(C-reactive protein)
	parhxmi = 1 if parental history of MI before age 60, 0 if not
	diaba1c = HbA1c if diabetic, 0 if not diabetic
	; 

*********************************;

** AHA/ACC model;
* For men;
if sex=1 then do;
if black=0 then do;
  bsurvrw=0.9144; 
  bxave_rw= 61.18;
  bxrw= 12.344*lnage+11.853*lnchol-2.664*lnchol*lnage-7.990*lnhdl+1.769*lnhdl*lnage
        +1.797*lsbp_trt+1.764*lsbp_untrt+7.837*currsmoke-1.795*currsmoke*lnage+0.658*diabetes;
end;
else if black=1 then do;
  bsurvrw=0.8954; 
  bxave_rw= 19.54;
  bxrw= 2.469*lnage+0.302*lnchol-0.307*lnhdl
        +1.916*lsbp_trt+1.809*lsbp_untrt+0.549*currsmoke+0.645*diabetes;
end;
survrw=bsurvrw**(exp(bxrw-bxave_rw));
probrw=1-survrw;
pctrw=100*probrw;
end;

** AHA/ACC model;
* For women;
else if sex=2 then do;
if black=0 then do;
  bsurvrw=0.9665; 
  bxave_rw= -29.18;
  bxrw=-29.799*lnage+4.884*lnage*lnage+13.540*lnchol-3.114*lnchol*lnage-13.578*lnhdl+3.149*lnhdl*lnage
        +2.019*lsbp_trt+1.957*lsbp_untrt+7.574*currsmoke-1.665*currsmoke*lnage+0.661*diabetes;
end;
else if black=1 then do;
  bsurvrw=0.9533; 
  bxave_rw= 86.61;
  bxrw= 17.114*lnage+0.940*lnchol-18.920*lnhdl+4.475*lnhdl*lnage+29.291*lsbp_trt
        -6.432*lsbp_trt*lnage+27.820*lsbp_untrt-6.087*lsbp_untrt*lnage+0.691*currsmoke+0.874*diabetes;
end;
survrw=bsurvrw**(exp(bxrw-bxave_rw));
probrw=1-survrw;
pctrw=100*probrw;
end;

*********************************;

** Original ATP III Model;
* For Men;
if sex=1 then do;
bsurvatp=0.9402;
bxave_atp=172.3002;
bxatp=52.00961*lnage+20.014077*lnchol-0.905964*lnhdl+1.305784*lnsbp
       +0.241549*trtsbp+12.096316*currsmoke-4.605038*lnchol*lnage-2.843670*currsmoke*lnage
       -2.933230*lnage*lnage;
survatp=bsurvatp**(exp(bxatp-bxave_atp));
probatp=1-survatp;
pctatp=100*probatp;
end;

* Original ATP III Model;
* For Women;
else if sex=2 then do;
bsurvatp=0.98767;
bxave_atp=146.5933;
bxatp=31.764*lnage+22.46521*lnchol-1.18773*lnhdl+2.552905*lnsbp
       +0.420251*trtsbp+13.07543*currsmoke-5.061*lnchol*lnage-2.99695*currsmoke*lnage;
survatp=bsurvatp**(exp(bxatp-bxave_atp));
probatp=1-survatp;
pctatp=100*probatp;
end;

*********************************;

** Reynolds Risk Score;
* For Men (non-diabetics only);
if sex=1 then do;
bsurv_rrs=0.8990;
bxave_rrs=33.097;
bxrrs=4.385*lnage+2.607*lnsbp+0.963*lnchol-0.772*lnhdl+0.405*currsmoke
       +0.102*lncrp+0.541*parhxmi;
survrrs=bsurv_rrs**(exp(bxrrs-bxave_rrs));
probrrs=1-survrrs;
pctrrs=100*probrrs;
end;

** Reynolds Risk Score;
* For Women (includes non-diabetics);
if sex=2 then do;
bsurv_rrs=0.989057; 
bxave_rrs=22.324591018;
bxrrs=0.07988*age+0.13360*diaba1c+3.13693*lnsbp+0.81795*currsmk+0.17989*lncrp
       +0.43770*famhxmi-1.17186*lnhdl+1.38163*lnchol;
survrrs=bsurv_rrs**(exp(bxrrs-bxave_rrs));
probrrs=1-survrrs;
pctrrs=100*probrrs;
end;

*********************************;
