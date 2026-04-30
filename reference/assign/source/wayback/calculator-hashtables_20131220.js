
/* mean(lowerbound,upperbound,male,female) */
 
var means = {"tc": new Array(new Array(25,39,5.5,5.11),new Array(40,49,5.77,5.64),new Array(50,59,6.02,6.29),new Array(60,90,5.72,6.37)),"hdl": new Array(new Array(25,39,1.35,1.58),new Array(40,49,1.37,1.64),new Array(50,59,1.4,1.71),new Array(60,90,1.36,1.66)),"cpd": new Array(new Array(25,39,13,14),new Array(40,49,17,16),new Array(50,59,20,14),new Array(60,90,16,15)),"sbp": new Array(new Array(25,39,128,116),new Array(40,49,130,121),new Array(50,59,133,131),new Array(60,90,141,139))}; 

/* beta(male,female) */

var betas = {"age": new Array(0.05698,0.07203),"tc": new Array(0.22286,0.1272),"hdl": new Array(-0.53684,-0.55836),"sbp": new Array(0.01183,0.01064),"diabetes": new Array(0.81558,0.97727),"chd": new Array(0.275,0.49159),"cpd": new Array(0.02005,0.02724),"simd": new Array(0.06296,0.09386)};
/* range(upperbound,lowerbound,description) */

var ranges = {"age": new Array(25,90,"Current age"),"tc": new Array(2,12.5,"Total cholesterol"),"hdl": new Array(0.3,3.5,"HDL Cholesterol"),"sbp": new Array(80,250,"Systolic Blood Pressure"),"cpd": new Array(0,100,"Cigarettes per day")};
	
/* notice(upperbound,lowerbound,description) */

var notices = {"age": new Array(30,74,"Current age"),"tc": new Array(3.5,9,"Total cholesterol"),"hdl": new Array(0.6,2.3,"HDL Cholesterol"),"sbp": new Array(100,200,"Systolic Blood Pressure"),"cpd": new Array(0,40,"Cigarettes per day")};
