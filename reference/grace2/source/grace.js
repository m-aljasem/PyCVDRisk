
//INPUT VARIABLES
//Age
var ageInput;
//Pulse
var pulseInput;
//Systolic blood pressure
var sbpInput;
//Creatine
var creatInput;
//Killip class
var killipInput;
//Cardiac arrest
var cardiacInput;
//Elevated troponin
var enzymesInput;
//ST deviation
var stemiInput;

//CALCULATED VARIABLES
//XBhat - using b in actual variable name instead of beta symbol
var xbhat_age;

//OTHER VARIABLES
var sZeroT = 0.9983577131;
var sZeroTSubstituted = 0.9994190637;
var numberOfDecimalPlaces = 12;

//----------------------------------- A. AGE ------------------------------------//
//Age variables
var a1_age = 4.8165;
var b1_age = 6.6571;
var c1_age = 8.3095;

//Age functions
function age_toDecades(age) {
    var ageInDecades = age / 10;
    return ageInDecades;
}

function age_calculateXbhat1(ageInDecades) {
    var linearTerm = ageInDecades * 0.41157;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function age_calculateXbhat1_substituded(ageInDecades) {
    var linearTerm = ageInDecades * 0.44230;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function age_f11(ageInDecades)
{
    var f11;
    if (ageInDecades >= a1_age) {
        f11 = Math.pow((ageInDecades - a1_age), 3);
    }
    else {
        f11 = 0;
    }
    return f11.roundup(numberOfDecimalPlaces);
}

function age_f12(ageInDecades) {
    var f12;
    if (ageInDecades >= b1_age) {
        var temp1 = Math.pow((ageInDecades - b1_age), 3);
        var temp2 = (c1_age - a1_age);
        var temp3 = (c1_age - b1_age)
        f12 = temp1 * temp2 / temp3  ;
    }
    else {
        f12 = 0;
    }
    return f12.roundup(numberOfDecimalPlaces);
}

function age_f13(ageInDecades) {
    var f13;
    if (ageInDecades >= c1_age) {
        var temp1 = Math.pow((ageInDecades - c1_age), 3);
        var temp2 = (b1_age - a1_age);
        var temp3 = (c1_age - b1_age)
        f13 = temp1 * temp2 / temp3;
    }
    else {
        f13 = 0;
    }
    return f13.roundup(numberOfDecimalPlaces);
}

function age_calculateXbhat2(ageInDecades)
{
    var x12_3;
    var f11 = age_f11(ageInDecades);
    var f12 = age_f12(ageInDecades);
    var f13 = age_f13(ageInDecades)
    x12_3 = parseFloat(f11) - parseFloat(f12) + parseFloat(f13);
    var xbhat2 = x12_3 * 0.01290;
    return xbhat2.roundup(numberOfDecimalPlaces);
}

function age_calculateXbhat2_substituded(ageInDecades) {
    var x12_3;
    var f11 = age_f11(ageInDecades);
    var f12 = age_f12(ageInDecades);
    var f13 = age_f13(ageInDecades)
    x12_3 = parseFloat(f11) - parseFloat(f12) + parseFloat(f13);
    var xbhat2 = x12_3 * 0.01231;
    return xbhat2.roundup(numberOfDecimalPlaces);
}

//----------MAIN AGE FUNCTIONS -------------
function age_calculateXbhat(age) {
    var ageInDecades = age_toDecades(age);
    var xbhat_1 = age_calculateXbhat1(ageInDecades);
    var xbhat_2 = age_calculateXbhat2(ageInDecades);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2);
    return xbhat.roundup(numberOfDecimalPlaces);
}

function age_calculateXbhat_substituted(age) {
    var ageInDecades = age_toDecades(age);
    var xbhat_1 = age_calculateXbhat1_substituded(ageInDecades);
    var xbhat_2 = age_calculateXbhat2_substituded(ageInDecades);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2);
    return xbhat.roundup(numberOfDecimalPlaces);
}

//----------------------------------- END OF A. AGE ------------------------------------//

//----------------------------------- B. PULSE -----------------------------------------//
//Pulse variables
var a3_pulse = 1.71;
var b3_pulse = 2.31;
var c3_pulse = 2.77;
var d3_pulse = 3.95

//Pulse functions
function pulse_to30BeatsPerMinute(pulse) {
    var pulseIn30BeatsPerMinute = pulse / 30;
    return pulseIn30BeatsPerMinute.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat1(pulseIn30BeatsPerMinute) {
    var linearTerm = pulseIn30BeatsPerMinute * 0.13138;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat1_substituted(pulseIn30BeatsPerMinute) {
    var linearTerm = pulseIn30BeatsPerMinute * 0.16705;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function pulse_f31(pulseIn30BeatsPerMinute) {
    var f31;
    if (pulseIn30BeatsPerMinute >= a3_pulse) {
        f31 = Math.pow((pulseIn30BeatsPerMinute - a3_pulse), 3);
    }
    else {
        f31 = 0;
    }
    return f31.roundup(numberOfDecimalPlaces);
}

function pulse_f32(pulseIn30BeatsPerMinute) {
    var f32;
    if (pulseIn30BeatsPerMinute >= c3_pulse) {
        var temp1 = Math.pow((pulseIn30BeatsPerMinute - c3_pulse), 3);
        var temp2 = (d3_pulse - a3_pulse);
        var temp3 = (d3_pulse - c3_pulse);
        var f32 = temp1 * temp2 / temp3
    }
    else {
        f32 = 0;
    }
    return f32.roundup(numberOfDecimalPlaces);
}

function pulse_f33(pulseIn30BeatsPerMinute) {
    var f33;
    if (pulseIn30BeatsPerMinute >= d3_pulse) {
        var temp1 = Math.pow((pulseIn30BeatsPerMinute - d3_pulse), 3);
        var temp2 = (c3_pulse - a3_pulse);
        var temp3 = (d3_pulse - c3_pulse);
        var f33 = temp1 * temp2 / temp3
    }
    else {
        f33 = 0;
    }
    return f33.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat2(pulseIn30BeatsPerMinute) {
    var x32_4;
    var f31 = pulse_f31(pulseIn30BeatsPerMinute);
    var f32 = pulse_f32(pulseIn30BeatsPerMinute);
    var f33 = pulse_f33(pulseIn30BeatsPerMinute);
    x32_4 = parseFloat(f31) - parseFloat(f32) + parseFloat(f33);
    var xbhat2 = x32_4 * 0.40176;
    return xbhat2.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat2_substituted(pulseIn30BeatsPerMinute) {
    var x32_4;
    var f31 = pulse_f31(pulseIn30BeatsPerMinute);
    var f32 = pulse_f32(pulseIn30BeatsPerMinute);
    var f33 = pulse_f33(pulseIn30BeatsPerMinute);
    x32_4 = parseFloat(f31) - parseFloat(f32) + parseFloat(f33);
    var xbhat2 = x32_4 * 0.43926;
    return xbhat2.roundup(numberOfDecimalPlaces);
}

function pulse_f34(pulseIn30BeatsPerMinute) {
    var f34;
    if (pulseIn30BeatsPerMinute >= b3_pulse) {
        var f34 = Math.pow((pulseIn30BeatsPerMinute - b3_pulse), 3);
    }
    else {
        f34 = 0;
    }
    return f34.roundup(numberOfDecimalPlaces);
}

function pulse_f35(pulseIn30BeatsPerMinute) {
    var f35;
    if (pulseIn30BeatsPerMinute >= c3_pulse) {
        var temp1 = Math.pow((pulseIn30BeatsPerMinute - c3_pulse), 3);
        var temp2 = (d3_pulse - b3_pulse);
        var temp3 = (d3_pulse - c3_pulse);
        var f35 = temp1 * temp2 / temp3
    }
    else {
        f35 = 0;
    }
    return f35.roundup(numberOfDecimalPlaces);
}

function pulse_f36(pulseIn30BeatsPerMinute) {
    var f36;
    if (pulseIn30BeatsPerMinute >= d3_pulse) {
        var temp1 = Math.pow((pulseIn30BeatsPerMinute - d3_pulse), 3);
        var temp2 = (c3_pulse - b3_pulse);
        var temp3 = (d3_pulse - c3_pulse);
        var f36 = temp1 * temp2 / temp3
    }
    else {
        f36 = 0;
    }
    return f36.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat3(pulseIn30BeatsPerMinute) {
    var x33_4;
    var f34 = pulse_f34(pulseIn30BeatsPerMinute);
    var f35 = pulse_f35(pulseIn30BeatsPerMinute);
    var f36 = pulse_f36(pulseIn30BeatsPerMinute);
    x33_4 = parseFloat(f34) - parseFloat(f35) + parseFloat(f36);
    var xbhat3 = x33_4 * -1.37249;
    return xbhat3.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat3_substituted(pulseIn30BeatsPerMinute) {
    var x33_4;
    var f34 = pulse_f34(pulseIn30BeatsPerMinute);
    var f35 = pulse_f35(pulseIn30BeatsPerMinute);
    var f36 = pulse_f36(pulseIn30BeatsPerMinute);
    x33_4 = parseFloat(f34) - parseFloat(f35) + parseFloat(f36);
    var xbhat3 = x33_4 * -1.48002;
    return xbhat3.roundup(numberOfDecimalPlaces);
}

//----------MAIN PULSE FUNCTIONS-------------
function pulse_calculateXbhat(pulse) {
    var pulse30Beats = pulse_to30BeatsPerMinute(pulse);
    var xbhat_1 = pulse_calculateXbhat1(pulse30Beats);
    var xbhat_2 = pulse_calculateXbhat2(pulse30Beats);
    var xbhat_3 = pulse_calculateXbhat3(pulse30Beats);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2) + parseFloat(xbhat_3);
    return xbhat.roundup(numberOfDecimalPlaces);
}

function pulse_calculateXbhat_substituted(pulse) {
    var pulse30Beats = pulse_to30BeatsPerMinute(pulse);
    var xbhat_1 = pulse_calculateXbhat1_substituted(pulse30Beats);
    var xbhat_2 = pulse_calculateXbhat2_substituted(pulse30Beats);
    var xbhat_3 = pulse_calculateXbhat3_substituted(pulse30Beats);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2) + parseFloat(xbhat_3);
    return xbhat.roundup(numberOfDecimalPlaces);
}

//----------------------------------- END OF B. PULSE ----------------------------------//

//------------------------------ C. SYSTOLIC BLOOD PRESSURE ----------------------------//

//SBP variables
var a2_sbp = -9.02;
var b2_sbp = -6.97;
var c2_sbp = -5.32;

//SBP functions
function sbp_toPerMinus22mmHg(sbp) {
    var sbpPerMinus22mmHg = sbp / -20;
    return sbpPerMinus22mmHg.roundup(numberOfDecimalPlaces);
}

function sbp_calculateXbhat1(sbpPerMinus22mmHg) {
    var linearTerm = sbpPerMinus22mmHg * 0.08222;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function sbp_calculateXbhat1_substituted(sbpPerMinus22mmHg) {
    var linearTerm = sbpPerMinus22mmHg * 0.02338;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function sbp_f21(sbpPerMinus22mmHg) {
    var f21;
    if (sbpPerMinus22mmHg >= a2_sbp) {
        f21 = Math.pow((sbpPerMinus22mmHg - a2_sbp), 3);
    }
    else {
        f21 = 0;
    }
    return f21.roundup(numberOfDecimalPlaces);
}

function sbp_f22(sbpPerMinus22mmHg) {
    var f22;
    if (sbpPerMinus22mmHg >= b2_sbp) {
        var temp1 = Math.pow((sbpPerMinus22mmHg - b2_sbp), 3);
        var temp2 = (c2_sbp - a2_sbp);
        var temp3 = (c2_sbp - b2_sbp)
        f22 = temp1 * temp2 / temp3;
    }
    else {
        f22 = 0;
    }
    return f22.roundup(numberOfDecimalPlaces);
}

function sbp_f23(sbpPerMinus22mmHg) {
    var f23;
    if (sbpPerMinus22mmHg >= c2_sbp) {
        var temp1 = Math.pow((sbpPerMinus22mmHg - c2_sbp), 3);
        var temp2 = (b2_sbp - a2_sbp);
        var temp3 = (c2_sbp - b2_sbp)
        f23 = temp1 * temp2 / temp3;
    }
    else {
        f23 = 0;
    }
    return f23.roundup(numberOfDecimalPlaces);
}

function sbp_calculateXbhat2(sbpPerMinus22mmHg) {
    var x22_3;
    var f21 = sbp_f21(sbpPerMinus22mmHg);
    var f22 = sbp_f22(sbpPerMinus22mmHg);
    var f23 = sbp_f23(sbpPerMinus22mmHg);
    x22_3 = parseFloat(f21) - parseFloat(f22) + parseFloat(f23);
    var xbhat2 = x22_3 * 0.01020;
    return xbhat2.roundup(numberOfDecimalPlaces);
}

function sbp_calculateXbhat2_substituted(sbpPerMinus22mmHg) {
    var x22_3;
    var f21 = sbp_f21(sbpPerMinus22mmHg);
    var f22 = sbp_f22(sbpPerMinus22mmHg);
    var f23 = sbp_f23(sbpPerMinus22mmHg);
    x22_3 = parseFloat(f21) - parseFloat(f22) + parseFloat(f23);
    var xbhat2 = x22_3 * 0.01835;
    return xbhat2.roundup(numberOfDecimalPlaces);
}

//----------MAIN SBP FUNCTION-------------
function sbp_calculateXbhat(sbp) {
    var sbpPerMinus22mmHg = sbp_toPerMinus22mmHg(sbp);
    var xbhat_1 = sbp_calculateXbhat1(sbpPerMinus22mmHg);
    var xbhat_2 = sbp_calculateXbhat2(sbpPerMinus22mmHg);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2);
    return xbhat.roundup(numberOfDecimalPlaces);
}

function sbp_calculateXbhat_substituted(sbp) {
    var sbpPerMinus22mmHg = sbp_toPerMinus22mmHg(sbp);
    var xbhat_1 = sbp_calculateXbhat1_substituted(sbpPerMinus22mmHg);
    var xbhat_2 = sbp_calculateXbhat2_substituted(sbpPerMinus22mmHg);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2);
    return xbhat.roundup(numberOfDecimalPlaces);
}

//-------------------------- END C. SYSTOLIC BLOOD PRESSURE ----------------------------//

//---------------------------------- D. CREATINE ---------------------------------------//

//Creatine variables
var a4_creatine = .685;
var b4_creatine = .925;
var c4_creatine = 1.145;
var d4_creatine = 2.065;

//Creatine functions
function creatine_calculateXbhat1(creatineLevel) {
    var linearTerm = creatineLevel * -0.51259;
    return linearTerm.roundup(numberOfDecimalPlaces);
}

function creatine_f41(creatineLevel) {
    var f41;
    if (creatineLevel >= a4_creatine) {
        f41 = Math.pow((creatineLevel - a4_creatine), 3);
    }
    else {
        f41 = 0;
    }
    return f41.roundup(numberOfDecimalPlaces);
}

function creatine_f42(creatineLevel) {
    var f42;
    if (creatineLevel >= c4_creatine) {
        var temp1 = Math.pow((creatineLevel - c4_creatine), 3);
        var temp2 = (d4_creatine - a4_creatine);
        var temp3 = (d4_creatine - c4_creatine)
        f42 = temp1 * temp2 / temp3;
    }
    else {
        f42 = 0;
    }
    return f42.roundup(numberOfDecimalPlaces);
}

function creatine_f43(creatineLevel) {
    var f43;
    if (creatineLevel >= d4_creatine) {
        var temp1 = Math.pow((creatineLevel - d4_creatine), 3);
        var temp2 = (c4_creatine - a4_creatine);
        var temp3 = (d4_creatine - c4_creatine)
        f43 = temp1 * temp2 / temp3;
    }
    else {
        f43 = 0;
    }
    return f43.roundup(numberOfDecimalPlaces);
}

function creatine_calculateXbhat2(creatineLevel) {
    var x42_4;
    var f41 = creatine_f41(creatineLevel);
    var f42 = creatine_f42(creatineLevel);
    var f43 = creatine_f43(creatineLevel);
    x42_4 = parseFloat(f41) - parseFloat(f42) + parseFloat(f43);
    var xbhat2 = x42_4 * 7.52634;
    if (creatineLevel == 1) {
        //alert(xbhat2);
    }
    return xbhat2.roundup(numberOfDecimalPlaces);
}

function creatine_f44(creatineLevel) {
    var f44;
    if (creatineLevel >= b4_creatine) {
        f44 = Math.pow((creatineLevel - b4_creatine), 3);
    }
    else {
        f44 = 0;
    }
    return f44.roundup(numberOfDecimalPlaces);
}

function creatine_f45(creatineLevel) {
    var f45;
    if (creatineLevel >= c4_creatine) {
        var temp1 = Math.pow((creatineLevel - c4_creatine), 3);
        var temp2 = (d4_creatine - b4_creatine);
        var temp3 = (d4_creatine - c4_creatine)
        f45 = temp1 * temp2 / temp3;
    }
    else {
        f45 = 0;
    }
    return f45.roundup(numberOfDecimalPlaces);
}

function creatine_f46(creatineLevel) {
    var f46;
    if (creatineLevel >= d4_creatine) {
        var temp1 = Math.pow((creatineLevel - d4_creatine), 3);
        var temp2 = (c4_creatine - b4_creatine);
        var temp3 = (d4_creatine - c4_creatine)
        f46 = temp1 * temp2 / temp3;
    }
    else {
        f46 = 0;
    }
    return f46.roundup(numberOfDecimalPlaces);
}

function creatine_calculateXbhat3(creatineLevel) {
    var x43_4;
    var f44 = creatine_f44(creatineLevel);
    var f45 = creatine_f45(creatineLevel);
    var f46 = creatine_f46(creatineLevel);
    x43_4 = parseFloat(f44) - parseFloat(f45) + parseFloat(f46);
    var xbhat3 = x43_4 * -18.23023;
    return xbhat3.roundup(numberOfDecimalPlaces);
}

//----------MAIN CREATINE FUNCTION-------------
function creatine_calculateXbhat(creatineLevel) {
    var xbhat_1 = creatine_calculateXbhat1(creatineLevel);
    var xbhat_2 = creatine_calculateXbhat2(creatineLevel);
    var xbhat_3 = creatine_calculateXbhat3(creatineLevel);
    var xbhat = parseFloat(xbhat_1) + parseFloat(xbhat_2) + parseFloat(xbhat_3);
    return xbhat.roundup(numberOfDecimalPlaces);
}

function creatine_calculateXbhat_substituted(renalInsufficiency) {
    if (renalInsufficiency) {
        return 0.44469;
    }
    else {
        return 0;
    }
}

//-------------------------------- END D. CREATINE --------------------------------------//

//------------------------------------ E. KILLIP ----------------------------------------//

//----------MAIN KILLOP FUNCTION-------------
function killip_calculateXbhat(killip) {
    killip = parseInt(killip);
    switch (killip) {
        case 1:
            return 0;
            break;
        case 2:
            return 0.63827;
            break;
        case 3:
            return 0.85325;
            break;
        case 4:
            return 1.29372;
            break;
    }
}

function killip_calculateXbhat_substituted(diureticUsage) {
    if (diureticUsage) {
        return 0.68008;
    }
    else {
        return 0;
    }
}


//-------------------------------- END E. KILLIP ---------------------------------------//

//------------------------------ F. CARDIAC ARREST -------------------------------------//

function cardiac_calculateXbhat(cardiac) {
    if (cardiac) {
        return 0.87185;
    }
    else {
        return 0;
    }
}

function cardiac_calculateXbhat_substituted(cardiac) {
    if (cardiac) {
        return 1.18193;
    }
    else {
        return 0;
    }
}


//----------------------------- END F. CARDIAC ARREST ----------------------------------//

//----------------------------------- G. ENZYMES ---------------------------------------//

function enzymes_calculateXbhat(enzymes) {
    if (enzymes) {
        return 0.37660;
    }
    else {
        return 0;
    }
}

function enzymes_calculateXbhat_substituted(enzymes) {
    if (enzymes) {
        return 0.39944;
    }
    else {
        return 0;
    }
}

//--------------------------------- END G. ENZYMES -------------------------------------//

//------------------------------------ H. STEMI ----------------------------------------//

function stemi_calculateXbhat(stemi) {
    if (stemi) {
        return 0.44303;
    }
    else {
        return 0;
    }
}

function stemi_calculateXbhat_substituted(stemi) {
    if (stemi) {
        return 0.46524;
    }
    else {
        return 0;
    }
}

//---------------------------------- END H. STEMI- -------------------------------------//


//--------------------------------- MAIN EQUATIONS -------------------------------------//

//------------------------- 1. DEATH AT ONE YEAR - LONG FORM ---------------------------//

function oneyear_longform(age, pulse, sbp, creatine, cardiac, enzymes, stemi, killip) {
    var xbhat_age = age_calculateXbhat(age);
    var xbhat_pulse = pulse_calculateXbhat(pulse);
    var xbhat_sbp = sbp_calculateXbhat(sbp);
    var xbhat_creatine = creatine_calculateXbhat(creatine);
    var xbhat_cardiac = cardiac_calculateXbhat(cardiac);
    var xbhat_enzymes = enzymes_calculateXbhat(enzymes);
    var xbhat_stemi = stemi_calculateXbhat(stemi);
    var xbhat_killip = killip_calculateXbhat(killip);

//    alert(xbhat_age);
//    alert(xbhat_pulse);
//    alert(xbhat_sbp);
//    alert(xbhat_creatine);
//    alert(xbhat_cardiac);
//    alert(xbhat_enzymes);
//    alert(xbhat_stemi);
//    alert(xbhat_killip);

    var xbhat = parseFloat(xbhat_age) + parseFloat(xbhat_pulse) + parseFloat(xbhat_sbp) + parseFloat(xbhat_creatine) + parseFloat(xbhat_cardiac) + parseFloat(xbhat_enzymes) + parseFloat(xbhat_stemi) + parseFloat(xbhat_killip);

    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    return result.roundup(2);
}

//------------------------- 2. DEATH AT ONE YEAR - SUBSTITUTED ---------------------------//

function oneyear_substituted(age, pulse, sbp, renalInsufficiency, cardiac, enzymes, stemi, diureticUsage) {
    var xbhat_age = age_calculateXbhat_substituted(age);
    var xbhat_pulse = pulse_calculateXbhat_substituted(pulse);
    var xbhat_sbp = sbp_calculateXbhat_substituted(sbp);
    var xbhat_creatine = creatine_calculateXbhat_substituted(renalInsufficiency);
    var xbhat_enzymes = enzymes_calculateXbhat_substituted(enzymes);
    var xbhat_stemi = stemi_calculateXbhat_substituted(stemi);
    var xbhat_killip = killip_calculateXbhat_substituted(diureticUsage);
    var xbhat_cardiac = cardiac_calculateXbhat_substituted(cardiac);

    var xbhat = parseFloat(xbhat_age) + parseFloat(xbhat_pulse) + parseFloat(xbhat_sbp) + parseFloat(xbhat_creatine) + parseFloat(xbhat_cardiac) + parseFloat(xbhat_enzymes) + parseFloat(xbhat_stemi) + parseFloat(xbhat_killip);

    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroTSubstituted, temp);
    var result = 1 - temp2;
    return result.roundup(2);
}

//------------------------------- END MAIN EQUATIONS -----------------------------------//

//--------------------------------- EXTERNAL API ---------------------------------------//

function calculate_oneyear(age, pulse, sbp, creatine, renalInsufficiency, cardiac, enzymes, stemi, killip, diureticUsage) {
    if (creatine == null || killip == null) {
        return oneyear_substituted(age, pulse, sbp, renalInsufficiency, cardiac, enzymes, stemi, diureticUsage);
    }
    else {
        //alert(age + ", " + pulse + ", " + sbp + ", " + creatine + ", " + cardiac + ", " + enzymes + ", " + stemi + ", " + killip);
        //alert("Killip" + killip);
        return oneyear_longform(age, pulse, sbp, creatine, cardiac, enzymes, stemi, killip);
    }
}

