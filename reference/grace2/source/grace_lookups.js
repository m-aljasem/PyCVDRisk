//toFixed is unreliable accross browsers so write our own
Number.prototype.roundup = function (dec) {
    if(this < 10)
    {
        var multiplier = Math.pow(10, dec);
        var result = (Math.round(this * multiplier) / multiplier).toFixed(dec);
        if(result.length == 1)
        {
            result = result + ".0";
        }
        return result;
    }
    else
    {
        return Math.round(this);
    }
}

//----------------------------------------------------------------------//
//                                                                      //
//                        In Hospital Death                             //
//                                                                      //
//----------------------------------------------------------------------//

var Admission_Age = {
    "29": "1.5399",
    "30": "1.593",
    "31": "1.6461",
    "32": "1.6992",
    "33": "1.7523",
    "34": "1.8054",
    "35": "1.8585",
    "36": "1.9116",
    "37": "1.9647",
    "38": "2.0178",
    "39": "2.0709",
    "40": "2.124",
    "41": "2.1771",
    "42": "2.2302",
    "43": "2.2833",
    "44": "2.3364",
    "45": "2.3895",
    "46": "2.4426",
    "47": "2.4957",
    "48": "2.5488",
    "49": "2.6019",
    "50": "2.655",
    "51": "2.7081",
    "52": "2.7612",
    "53": "2.8143",
    "54": "2.8674",
    "55": "2.9205",
    "56": "2.9736",
    "57": "3.0267",
    "58": "3.0798",
    "59": "3.1329",
    "60": "3.186",
    "61": "3.2391",
    "62": "3.2922",
    "63": "3.3453",
    "64": "3.3984",
    "65": "3.4515",
    "66": "3.5046",
    "67": "3.5577",
    "68": "3.6108",
    "69": "3.6639",
    "70": "3.717",
    "71": "3.7701",
    "72": "3.8232",
    "73": "3.8763",
    "74": "3.9294",
    "75": "3.9825",
    "76": "4.0356",
    "77": "4.0887",
    "78": "4.1418",
    "79": "4.1949",
    "80": "4.248",
    "81": "4.3011",
    "82": "4.3542",
    "83": "4.4073",
    "84": "4.4604",
    "85": "4.5135",
    "86": "4.5666",
    "87": "4.6197",
    "88": "4.6728",
    "89": "4.7259",
    "90": "4.779",
    "91": "4.8321",
    "92": "4.8852",
    "93": "4.9383",
    "94": "4.9914",
    "95": "5.0445",
    "96": "5.0976",
    "97": "5.1507",
    "98": "5.2038",
    "99": "5.2569",
    "100": "5.31"
};

var Admission_Pulse = {
    "49":"0.4263",
    "59.5":"0.51765",
    "74.5":"0.64815",
    "84.5":"0.73515",
    "94.5":"0.82215",
    "104.5":"0.90915",
    "119.5":"1.03965",
    "139.5":"1.21365",
    "174.5":"1.51815",
    "200":"1.74"
};

var Admission_BloodPressure = {
    "79":"-1.3272",
    "89.5":"-1.5036",
    "104.5":"-1.7556",
    "114.5":"-1.9236",
    "124.5":"-2.0916",
    "134.5":"-2.2596",
    "149.5":"-2.5116",
    "169.5":"-2.8476",
    "180":"-3.024"
};

var Admission_Creatine =
{
    "0.195":"0.0355485",
    "0.595":"0.1084685",
    "0.995":"0.1813885",
    "1.395":"0.2543085",
    "1.795":"0.3272285",
    "2.995":"0.5459885",
    "4.0":"0.7292"
};

var Admission_Renal =
{
    "no": "0.09115",
    "yes": "0.3646"
};

var Admission_Killip =
{
    "1": "0.6931",
    "2": "1.3862",
    "3": "2.0793",
    "4": "2.7724"
};

var Admission_Diuretic =
{
    "no": "0.6931",
    "yes": "1.73275"
};

var Admission_Cardiac =
{
    "no": "0",
    "yes": "1.4586"
};

var Admission_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0.4700"
};

var Admission_STEMI =
{
    "no": "0",
    "yes": "0.8755"
};

function CalculateAdmissionToInHospitalDeath(age, pulse, bloodPressure, creatinine, renal, killip, diauretic, cardiac, enzymes, stemi) {
    var creatinineValue;
    var killipValue;
    var fiddleFactor = -7.7035;
    var ageValue = Admission_Age[age];
    var pulseValue = Admission_Pulse[pulse];
    var bloodPressureValue = Admission_BloodPressure[bloodPressure];

    if (creatinine != 'na' && creatinine != 'not') {
        creatinineValue = Admission_Creatine[creatinine];
    }
    else {
        creatinineValue = Admission_Renal[renal];
    }
    if (killip != 'na') {
        killipValue = Admission_Killip[killip];
    }
    else {
        killipValue = Admission_Diuretic[diauretic];
    }
    var cardiacValue = Admission_Cardiac[cardiac];
    var enzymesValue = Admission_ElevatedEnzymes[enzymes];
    var stemiValue = Admission_STEMI[stemi];
    //alert(fiddleFactor + "," + ageValue + "," + pulseValue + "," + bloodPressureValue + "," + creatinineValue  + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue); 
    var xbhat = parseFloat(fiddleFactor) + parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);

    var temp = Math.exp(xbhat);

    var probabilityOfDeath = temp / (1 + temp);

    probabilityOfDeath = probabilityOfDeath * 100;
    

    return parseFloat(probabilityOfDeath.roundup(1));
}


//----------------------------------------------------------------------//
//                                                                      //
//                    In Hospital Death / MI                            //
//                                                                      //
//----------------------------------------------------------------------//


var Admission_MI_Age = {
    "29": "0",
    "30": "0",
    "31": "1.3",
    "32": "2.6",
    "33": "3.9",
    "34": "5.2",
    "35": "6.5",
    "36": "7.8",
    "37": "9.1",
    "38": "10.4",
    "39": "11.7",
    "40": "13",
    "41": "14.4",
    "42": "15.8",
    "43": "17.2",
    "44": "18.6",
    "45": "20",
    "46": "21.4",
    "47": "22.8",
    "48": "24.2",
    "49": "25.6",
    "50": "27",
    "51": "28.3",
    "52": "29.6",
    "53": "30.9",
    "54": "32.2",
    "55": "33.5",
    "56": "34.8",
    "57": "36.1",
    "58": "37.4",
    "59": "38.7",
    "60": "40",
    "61": "41.4",
    "62": "42.8",
    "63": "44.2",
    "64": "45.6",
    "65": "47",
    "66": "48.4",
    "67": "49.8",
    "68": "51.2",
    "69": "52.6",
    "70": "54",
    "71": "55.3",
    "72": "56.6",
    "73": "57.9",
    "74": "59.2",
    "75": "60.5",
    "76": "61.8",
    "77": "63.1",
    "78": "64.4",
    "79": "65.7",
    "80": "67",
    "81": "68.3",
    "82": "69.6",
    "83": "70.9",
    "84": "72.2",
    "85": "73.5",
    "86": "74.8",
    "87": "76.1",
    "88": "77.4",
    "89": "78.7",
    "90": "80",
    "91": "80",
    "92": "80",
    "93": "80",
    "94": "80",
    "95": "80",
    "96": "80",
    "97": "80",
    "98": "80",
    "99": "80",
    "100": "80"
};

var Admission_MI_Pulse = {
    "49":"0",
    "59.5":"2.85",
    "74.5":"6.35",
    "84.5":"9.35",
    "94.5":"12.35",
    "104.5":"14.9",
    "119.5":"18.85",
    "139.5":"24.85",
    "174.5":"34.37",
    "200":"41"
};

var Admission_MI_BloodPressure = {
    "79":"53.",
    "89.5":"49.2",
    "104.5":"42.75",
    "114.5":"37.75",
    "124.5":"33.2",
    "134.5":"28.75",
    "149.5":"22.2",
    "169.5":"13.2",
    "180":"9.0"
};

var Admission_MI_Creatine =
{
    "0.195": "0.975",
    "0.595": "3.975",
    "0.995": "6.975",
    "1.395": "9.95",
    "1.795": "12.95",
    "2.995": "20.965",
    "4.0": "28"
};

var Admission_MI_Renal =
{
    "no": "3.5",
    "yes": "14"
};

var Admission_MI_Killip =
{
    "1": "0",
    "2": "33",
    "3": "67",
    "4": "100"
};

var Admission_MI_Diuretic =
{
    "no": "0",
    "yes": "50"
};

var Admission_MI_Cardiac =
{
    "no": "0",
    "yes": "1"
};

var Admission_MI_ElevatedEnzymes =
{
    "no": "0",
    "yes": "1"
};

var Admission_MI_STEMI =
{
    "no": "0",
    "yes": "1"
};

function CalculateAdmissionToInHospitalDeathMI(age, pulse, bloodPressure, creatinine, renal, killip, diauretic, cardiac, enzymes, stemi) {

    var creatinineValue;
    var killipValue;
    var ageValue = Admission_MI_Age[age];
    var pulseValue = Admission_MI_Pulse[pulse];
    var bloodPressureValue = Admission_MI_BloodPressure[bloodPressure];

    if (creatinine != 'na' && creatinine != 'not') {
        creatinineValue = Admission_MI_Creatine[creatinine];
    }
    else {
        creatinineValue = Admission_MI_Renal[renal];
    }
    if (killip != 'na') {
        killipValue = Admission_MI_Killip[killip];
    }
    else {
        killipValue = Admission_MI_Diuretic[diauretic];
    }
    var cardiacValue = Admission_MI_Cardiac[cardiac];
    var enzymesValue = Admission_MI_ElevatedEnzymes[enzymes];
    var stemiValue = Admission_MI_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + creatinineValue  + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue); 
    //console.log(killipValue + "," + bloodPressureValue  + "," + pulseValue  + "," + ageValue  + "," + creatinineValue  + "," + stemiValue  + "," + enzymesValue  + "," + cardiacValue);
    var deathmi_pt =  parseFloat(killipValue) + parseFloat(bloodPressureValue) + parseFloat(pulseValue) + parseFloat(ageValue) + parseFloat(creatinineValue) + parseFloat((67 * stemiValue)) + parseFloat((54 * enzymesValue)) + parseFloat((98 * cardiacValue));
    var probability;

    if (deathmi_pt < 30) {
        probability = 0.02;
    }
    else if (deathmi_pt >= 30 && deathmi_pt <= 56)
    {
        probability = 0.03;
    }
    else if (deathmi_pt > 56 && deathmi_pt <= 78) {
        probability = 0.04;
    }
    else if (deathmi_pt > 78 && deathmi_pt <= 95) {
        probability = 0.05;
    }
    else if (deathmi_pt > 95 && deathmi_pt <= 110) {
        probability = 0.06;
    }
    else if (deathmi_pt > 110 && deathmi_pt <= 123) {
        probability = 0.07;
    }
    else if (deathmi_pt > 123 && deathmi_pt <= 135) {
        probability = 0.08;
    }
    else if (deathmi_pt > 135 && deathmi_pt <= 145) {
        probability = 0.09;
    }
    else if (deathmi_pt > 145 && deathmi_pt <= 155) {
        probability = 0.10;
    }
    else if (deathmi_pt > 155 && deathmi_pt <= 164) {
        probability = 0.11;
    }
    else if (deathmi_pt > 164 && deathmi_pt <= 172) {
        probability = 0.12;
    }
    else if (deathmi_pt > 172 && deathmi_pt <= 180) {
        probability = 0.13;
    }
    else if (deathmi_pt > 180 && deathmi_pt <= 187) {
        probability = 0.14;
    }
    else if (deathmi_pt > 187 && deathmi_pt <= 194) {
        probability = 0.15;
    }
    else if (deathmi_pt > 194 && deathmi_pt <= 201) {
        probability = 0.16;
    }
    else if (deathmi_pt > 201 && deathmi_pt <= 207) {
        probability = 0.17;
    }
    else if (deathmi_pt > 207 && deathmi_pt <= 213) {
        probability = 0.18;
    }
    else if (deathmi_pt > 213 && deathmi_pt <= 219) {
        probability = 0.19;
    }
    else if (deathmi_pt > 219 && deathmi_pt <= 224) {
        probability = 0.20;
    }
    else if (deathmi_pt > 224 && deathmi_pt <= 230) {
        probability = 0.21;
    }
    else if (deathmi_pt > 230 && deathmi_pt <= 235) {
        probability = 0.22;
    }
    else if (deathmi_pt > 235 && deathmi_pt <= 240) {
        probability = 0.23;
    }
    else if (deathmi_pt > 240 && deathmi_pt <= 245) {
        probability = 0.24;
    }
    else if (deathmi_pt > 245 && deathmi_pt <= 250) {
        probability = 0.25;
    }
    else if (deathmi_pt > 250 && deathmi_pt <= 254) {
        probability = 0.26;
    }
    else if (deathmi_pt > 254 && deathmi_pt <= 259) {
        probability = 0.27;
    }
    else if (deathmi_pt > 259 && deathmi_pt <= 263) {
        probability = 0.28;
    }
    else if (deathmi_pt > 263 && deathmi_pt <= 268) {
        probability = 0.29;
    }
    else if (deathmi_pt > 268 && deathmi_pt <= 308) {
        probability = 0.30;
    }
    else if (deathmi_pt > 308 && deathmi_pt <= 344) {
        probability = 0.40;
    }
    else if (deathmi_pt > 344 && deathmi_pt <= 381) {
        probability = 0.50;
    }
    else if (deathmi_pt > 381 && deathmi_pt <= 421) {
        probability = 0.60;
    }
    else if (deathmi_pt > 421 && deathmi_pt <= 470) {
        probability = 0.70;
    }
    else if (deathmi_pt > 470) {
        probability = 0.80;
    }
    probability = probability * 100;
    var toReturn = new Array();
    toReturn[0] = parseFloat(probability.roundup(1));
    toReturn[1] = deathmi_pt;
    return toReturn;
}


//----------------------------------------------------------------------//
//                                                                      //
//                           6 Month Death                              //
//                                                                      //
//----------------------------------------------------------------------//

var SixMonth_Age = {
    "29": "0",
    "30": "0",
    "31": "0",
    "32": "0",
    "33": "0",
    "34": "0",
    "35": "0",
    "36": "1.8",
    "37": "3.6",
    "38": "5.4",
    "39": "7.2",
    "40": "9",
    "41": "10.8",
    "42": "12.6",
    "43": "14.4",
    "44": "16.2",
    "45": "18",
    "46": "19.8",
    "47": "21.6",
    "48": "23.4",
    "49": "25.2",
    "50": "27",
    "51": "28.8",
    "52": "30.6",
    "53": "32.4",
    "54": "34.2",
    "55": "36",
    "56": "37.8",
    "57": "39.6",
    "58": "41.4",
    "59": "43.2",
    "60": "45",
    "61": "46.8",
    "62": "48.6",
    "63": "50.4",
    "64": "52.2",
    "65": "54",
    "66": "55.9",
    "67": "57.8",
    "68": "59.7",
    "69": "61.6",
    "70": "63.5",
    "71": "65.4",
    "72": "67.3",
    "73": "69.2",
    "74": "71.1",
    "75": "73",
    "76": "74.8",
    "77": "76.6",
    "78": "78.4",
    "79": "80.2",
    "80": "82",
    "81": "83.8",
    "82": "85.6",
    "83": "87.4",
    "84": "89.2",
    "85": "91",
    "86": "92.8",
    "87": "94.6",
    "88": "96.4",
    "89": "98.2",
    "90": "100",
    "91": "100",
    "92": "100",
    "93": "100",
    "94": "100",
    "95": "100",
    "96": "100",
    "97": "100",
    "98": "100",
    "99": "100",
    "100": "100"
};

var SixMonth_Pulse = {
    "49":"0",
    "59.5":"0",
    "74.5":"1.35",
    "84.5":"3.9",
    "94.5":"6.35",
    "104.5":"8.9",
    "119.5":"12.85",
    "139.5":"18.85",
    "174.5":"34.37",
    "200":"34.0"
};

var SixMonth_BloodPressure = {
    "79":"40",
    "89.5":"37.15",
    "104.5":"32.65",
    "114.5":"29.2",
    "124.5":"25.65",
    "134.5":"22.65",
    "149.5":"16.2",
    "169.5":"11.15",
    "180":"8.0"
};

var SixMonth_Creatine =
{
    "0.195": "0.975",
    "0.595": "3.975",
    "0.995": "6.975",
    "1.395": "9.95",
    "1.795": "12.95",
    "2.995": "20.965",
    "4.0":   "28"
};

var SixMonth_Renal =
{
    "no": "3.5",
    "yes": "14"
};

var SixMonth_Killip =
{
    "1": "0",
    "2": "15",
    "3": "29",
    "4": "44"
};

var SixMonth_Diuretic =
{
    "no": "0",
    "yes": "20"
};

var SixMonth_Cardiac =
{
    "no": "0",
    "yes": "1"
};

var SixMonth_ElevatedEnzymes =
{
    "no": "0",
    "yes": "1"
};

var SixMonth_STEMI =
{
    "no": "0",
    "yes": "1"
};

function CalculateSixMonthDeath(age, pulse, bloodPressure, creatinine, renal, killip, diauretic, cardiac, enzymes, stemi) {

    var creatinineValue;
    var killipValue;
    var ageValue = SixMonth_Age[age];
    var pulseValue = SixMonth_Pulse[pulse];
    var bloodPressureValue = SixMonth_BloodPressure[bloodPressure];

    if (creatinine != 'na' && creatinine != 'not') {
        creatinineValue = SixMonth_Creatine[creatinine];
    }
    else {
        creatinineValue = SixMonth_Renal[renal];
    }
    if (killip != 'na') {
        killipValue = SixMonth_Killip[killip];
    }
    else {
        killipValue = SixMonth_Diuretic[diauretic];
    }
    var cardiacValue = SixMonth_Cardiac[cardiac];
    var enzymesValue = SixMonth_ElevatedEnzymes[enzymes];
    var stemiValue = SixMonth_STEMI[stemi];

    var deathmi_pt = parseFloat(killipValue) + parseFloat(bloodPressureValue) + parseFloat(pulseValue) + parseFloat(ageValue) + parseFloat(creatinineValue) + parseFloat((17 * stemiValue)) + parseFloat((13 * enzymesValue)) + parseFloat((30 * cardiacValue));
    var probability;

    if (deathmi_pt <= 6) {
        probability = 0.002;
    }
    else if (deathmi_pt > 7 && deathmi_pt <= 27) {
        probability = 0.004;
    }
    else if (deathmi_pt > 27 && deathmi_pt <= 39) {
        probability = 0.006;
    }
    else if (deathmi_pt > 39 && deathmi_pt <= 48) {
        probability = 0.008;
    }
    else if (deathmi_pt > 48 && deathmi_pt <= 55) {
        probability = 0.01;
    }
    else if (deathmi_pt > 55 && deathmi_pt <= 60) {
        probability = 0.012;
    }
    else if (deathmi_pt > 60 && deathmi_pt <= 65) {
        probability = 0.014;
    }
    else if (deathmi_pt > 65 && deathmi_pt <= 69) {
        probability = 0.016;
    }
    else if (deathmi_pt > 69 && deathmi_pt <= 73) {
        probability = 0.018;
    }
    else if (deathmi_pt > 73 && deathmi_pt <= 76) {
        probability = 0.02;
    }
    else if (deathmi_pt > 76 && deathmi_pt <= 88) {
        probability = 0.03;
    }
    else if (deathmi_pt > 88 && deathmi_pt <= 97) {
        probability = 0.04;
    }
    else if (deathmi_pt > 97 && deathmi_pt <= 104) {
        probability = 0.05;
    }
    else if (deathmi_pt > 104 && deathmi_pt <= 110) {
        probability = 0.06;
    }
    else if (deathmi_pt > 110 && deathmi_pt <= 115) {
        probability = 0.07;
    }
    else if (deathmi_pt > 115 && deathmi_pt <= 119) {
        probability = 0.08;
    }
    else if (deathmi_pt > 119 && deathmi_pt <= 123) {
        probability = 0.09;
    }
    else if (deathmi_pt > 123 && deathmi_pt <= 126) {
        probability = 0.10;
    }
    else if (deathmi_pt > 126 && deathmi_pt <= 129) {
        probability = 0.11;
    }
    else if (deathmi_pt > 129 && deathmi_pt <= 132) {
        probability = 0.12;
    }
    else if (deathmi_pt > 132 && deathmi_pt <= 134) {
        probability = 0.13;
    }
    else if (deathmi_pt > 134 && deathmi_pt <= 137) {
        probability = 0.14;
    }
    else if (deathmi_pt > 137 && deathmi_pt <= 139) {
        probability = 0.15;
    }
    else if (deathmi_pt > 139 && deathmi_pt <= 141) {
        probability = 0.16;
    }
    else if (deathmi_pt > 141 && deathmi_pt <= 143) {
        probability = 0.17;
    }
    else if (deathmi_pt > 143 && deathmi_pt <= 146) {
        probability = 0.18;
    }
    else if (deathmi_pt > 146 && deathmi_pt <= 147) {
        probability = 0.19;
    }
    else if (deathmi_pt > 147 && deathmi_pt <= 149) {
        probability = 0.20;
    }
    else if (deathmi_pt > 149 && deathmi_pt <= 150) {
        probability = 0.21;
    }
    else if (deathmi_pt > 150 && deathmi_pt <= 152) {
        probability = 0.22;
    }
    else if (deathmi_pt > 152 && deathmi_pt <= 153) {
        probability = 0.23;
    }
    else if (deathmi_pt > 153 && deathmi_pt <= 155) {
        probability = 0.24;
    }
    else if (deathmi_pt > 155 && deathmi_pt <= 156) {
        probability = 0.25;
    }
    else if (deathmi_pt > 156 && deathmi_pt <= 158) {
        probability = 0.26;
    }
    else if (deathmi_pt > 158 && deathmi_pt <= 159) {
        probability = 0.27;
    }
    else if (deathmi_pt > 159 && deathmi_pt <= 160) {
        probability = 0.28;
    }
    else if (deathmi_pt > 160 && deathmi_pt <= 162) {
        probability = 0.29;
    }
    else if (deathmi_pt > 162 && deathmi_pt <= 163) {
        probability = 0.3;
    }
    else if (deathmi_pt > 163 && deathmi_pt <= 174) {
        probability = 0.4;
    }
    else if (deathmi_pt > 174 && deathmi_pt <= 183) {
        probability = 0.5;
    }
    else if (deathmi_pt > 183 && deathmi_pt <= 191) {
        probability = 0.6;
    }
    else if (deathmi_pt > 191 && deathmi_pt <= 200) {
        probability = 0.7;
    }
    else if (deathmi_pt > 200 && deathmi_pt <= 208) {
        probability = 0.8;
    }
    else if (deathmi_pt > 208 && deathmi_pt <= 219) {
        probability = 0.9;
    }
    else if (deathmi_pt > 219 && deathmi_pt <= 240) {
        probability = 0.99;
    }
    else if (deathmi_pt > 240) {
        probability = 1.13;
    }

    probability = probability * 100;

    var uncorrected = probability.roundup(1);

    probability = probability * (80 / 91);

    if (probability > 90) {
        probability = 92;
    }

    var toReturn = new Array();
    toReturn[0] = parseFloat(probability.roundup(1));
    toReturn[1] = deathmi_pt;
    toReturn[2] = uncorrected;

    return toReturn;
}

//----------------------------------------------------------------------//
//                                                                      //
//                      6 Month Death / MI                              //
//                                                                      //
//----------------------------------------------------------------------//

var SixMonth_MI_Age = {
    "29": "0",
    "30": "0",
    "31": "0",
    "32": "0",
    "33": "0",
    "34": "0",
    "35": "0",
    "36": "1.8",
    "37": "3.6",
    "38": "5.4",
    "39": "7.2",
    "40": "9",
    "41": "10.8",
    "42": "12.6",
    "43": "14.4",
    "44": "16.2",
    "45": "18",
    "46": "19.8",
    "47": "21.6",
    "48": "23.4",
    "49": "25.2",
    "50": "27",
    "51": "28.8",
    "52": "30.6",
    "53": "32.4",
    "54": "34.2",
    "55": "36",
    "56": "37.8",
    "57": "39.6",
    "58": "41.4",
    "59": "43.2",
    "60": "45",
    "61": "46.8",
    "62": "48.6",
    "63": "50.4",
    "64": "52.2",
    "65": "54",
    "66": "55.9",
    "67": "57.8",
    "68": "59.7",
    "69": "61.6",
    "70": "63.5",
    "71": "65.4",
    "72": "67.3",
    "73": "69.2",
    "74": "71.1",
    "75": "73",
    "76": "74.8",
    "77": "76.6",
    "78": "78.4",
    "79": "80.2",
    "80": "82",
    "81": "83.8",
    "82": "85.6",
    "83": "87.4",
    "84": "89.2",
    "85": "91",
    "86": "92.8",
    "87": "94.6",
    "88": "96.4",
    "89": "98.2",
    "90": "100",
    "91": "100",
    "92": "100",
    "93": "100",
    "94": "100",
    "95": "100",
    "96": "100",
    "97": "100",
    "98": "100",
    "99": "100",
    "100": "100"
};

var SixMonth_MI_BloodPressure = {
    "79":"54",
    "89.5":"49.725",
    "104.5":"42.975",
    "114.5":"38.475",
    "124.5":"33.975",
    "134.5":"29.475",
    "149.5":"22.725",
    "169.5":"13.725",
    "180":"9"
};

var SixMonth_MI_Creatine =
{
    "0.195": "1.95",
    "0.595": "5.95",
    "0.995": "9.95",
    "1.395": "13.95",
    "1.795": "17.95",
    "2.995": "29.95",
    "4.0":   "41"
};

var SixMonth_MI_Renal =
{
    "no": "5",
    "yes": "19"
};

var SixMonth_MI_Killip =
{
    "1": "0",
    "2": "27",
    "3": "55",
    "4": "82"
};

var SixMonth_MI_Diuretic =
{
    "no": "0",
    "yes": "40"
};

var SixMonth_MI_Cardiac =
{
    "no": "0",
    "yes": "1"
};

var SixMonth_MI_ElevatedEnzymes =
{
    "no": "0",
    "yes": "1"
};

var SixMonth_MI_STEMI =
{
    "no": "0",
    "yes": "1"
};

function CalculateSixMonthDeathMI(age, pulse, bloodPressure, creatinine, renal, killip, diauretic, cardiac, enzymes, stemi) {

    var creatinineValue;
    var killipValue;
    var ageValue = SixMonth_MI_Age[age];
    var bloodPressureValue = SixMonth_MI_BloodPressure[bloodPressure];

    if (creatinine != 'na' && creatinine != 'not') {
        creatinineValue = SixMonth_MI_Creatine[creatinine];
    }
    else {
        creatinineValue = SixMonth_MI_Renal[renal];
    }
    if (killip != 'na') {
        killipValue = SixMonth_MI_Killip[killip];
    }
    else {
        killipValue = SixMonth_MI_Diuretic[diauretic];
    }
    var cardiacValue = SixMonth_MI_Cardiac[cardiac];
    var enzymesValue = SixMonth_MI_ElevatedEnzymes[enzymes];
    var stemiValue = SixMonth_MI_STEMI[stemi];

    //alert(killipValue + "," + bloodPressureValue + "," + ageValue + "," + creatinineValue + "," + stemiValue + "," + enzymesValue + "," + cardiacValue);

    var deathmi_pt = parseFloat(killipValue) + parseFloat(bloodPressureValue) + parseFloat(ageValue) + parseFloat(creatinineValue) + parseFloat((39 * stemiValue)) + parseFloat((41 * enzymesValue)) + parseFloat((66 * cardiacValue));
    //alert(deathmi_pt);
    var probability;

    if (deathmi_pt <= 2) {
        probability = 0.04;
    }
    else if (deathmi_pt > 3 && deathmi_pt <= 20) {
        probability = 0.05;
    }
    else if (deathmi_pt > 20 && deathmi_pt <= 38) {
        probability = 0.06;
    }
    else if (deathmi_pt > 38 && deathmi_pt <= 46) {
        probability = 0.06;
    }
    else if (deathmi_pt > 46 && deathmi_pt <= 50) {
        probability = 0.07;
    }
    else if (deathmi_pt > 50 && deathmi_pt <= 55) {
        probability = 0.07;
    }
    else if (deathmi_pt > 55 && deathmi_pt <= 67) {
        probability = 0.08;
    }
    else if (deathmi_pt > 67 && deathmi_pt <= 74) {
        probability = 0.09;
    }
    else if (deathmi_pt > 74 && deathmi_pt <= 77) {
        probability = 0.10;
    }
    else if (deathmi_pt > 77 && deathmi_pt <= 86) {
        probability = 0.11;
    }
    else if (deathmi_pt > 86 && deathmi_pt <= 96) {
        probability = 0.12;
    }
    else if (deathmi_pt > 96 && deathmi_pt <= 104) {
        probability = 0.13;
    }
    else if (deathmi_pt > 104 && deathmi_pt <= 105) {
        probability = 0.14;
    }
    else if (deathmi_pt > 105 && deathmi_pt <= 114) {
        probability = 0.15;
    }
    else if (deathmi_pt > 114 && deathmi_pt <= 122) {
        probability = 0.17;
    }
    else if (deathmi_pt > 122 && deathmi_pt <= 132) {
        probability = 0.19;
    }
    else if (deathmi_pt > 132 && deathmi_pt <= 141) {
        probability = 0.21;
    }
    else if (deathmi_pt > 141 && deathmi_pt <= 148) {
        probability = 0.23;
    }
    else if (deathmi_pt > 148 && deathmi_pt <= 165) {
        probability = 0.27;
    }
    else if (deathmi_pt > 165 && deathmi_pt <= 168) {
        probability = 0.28;
    }
    else if (deathmi_pt > 168 && deathmi_pt <= 187) {
        probability = 0.34;
    }
    else if (deathmi_pt > 187 && deathmi_pt <= 196) {
        probability = 0.38;
    }
    else if (deathmi_pt > 196 && deathmi_pt <= 205) {
        probability = 0.40;
    }
    else if (deathmi_pt > 205 && deathmi_pt <= 223) {
        probability = 0.40;
    }
    else if (deathmi_pt > 223 && deathmi_pt <= 232) {
        probability = 0.50;
    }
    else if (deathmi_pt > 232 && deathmi_pt <= 242) {
        probability = 0.50;
    }
    else if (deathmi_pt > 242 && deathmi_pt <= 260) {
        probability = 0.60;
    }
    else if (deathmi_pt > 260 && deathmi_pt <= 271) {
        probability = 0.70;
    }
    else if (deathmi_pt > 271 && deathmi_pt <= 287) {
        probability = 0.70;
    }
    else if (deathmi_pt > 287 && deathmi_pt <= 296) {
        probability = 0.80;
    }
    else if (deathmi_pt > 296 && deathmi_pt <= 309) {
        probability = 0.80;
    }
    else if (deathmi_pt > 309 && deathmi_pt <= 327) {
        probability = 0.90;
    }
    else if (deathmi_pt > 327) {
        probability = 0.91;
    }

    probability = probability * 100;
    
    var unadjusted = probability.roundup(1);

    probability = probability * (120 / 158);

    var toReturn = new Array();
    toReturn[0] = parseFloat(probability.roundup(1));
    toReturn[1] = deathmi_pt;
    toReturn[2] = unadjusted;

    return toReturn;
}

//----------------------------------------------------------------------//
//                                                                      //
//                            1 Year Death                              //
//                                                                      //
//----------------------------------------------------------------------//

var OneYear_Age = {
    "29":"1.193553",
    "30":"1.23471",
    "31":"1.275867",
    "32":"1.317024",
    "33":"1.358181",
    "34":"1.399338",
    "35":"1.440495",
    "36":"1.481652",
    "37":"1.522809",
    "38":"1.563966",
    "39":"1.605123",
    "40":"1.64628",
    "41":"1.687437",
    "42":"1.728594",
    "43":"1.769751",
    "44":"1.810908",
    "45":"1.852065",
    "46":"1.893222",
    "47":"1.934379",
    "48":"1.975536",
    "49":"2.0167005",
    "50":"2.0579297",
    "51":"2.0993009",
    "52":"2.1408916",
    "53":"2.1827791",
    "54":"2.2250408",
    "55":"2.2677541",
    "56":"2.3109965",
    "57":"2.3548453",
    "58":"2.3993779",
    "59":"2.4446718",
    "60":"2.4908043",
    "61":"2.5378528",
    "62":"2.5858947",
    "63":"2.6350075",
    "64":"2.6852685",
    "65":"2.7367551",
    "66":"2.7895448",
    "67":"2.8437127",
    "68":"2.8992632",
    "69":"2.9561152",
    "70":"3.0141823",
    "71":"3.0733784",
    "72":"3.1336173",
    "73":"3.1948128",
    "74":"3.2568786",
    "75":"3.3197285",
    "76":"3.3832763",
    "77":"3.4474359",
    "78":"3.5121209",
    "79":"3.5772452",
    "80":"3.6427226",
    "81":"3.7084667",
    "82":"3.7743915",
    "83":"3.8404108",
    "84":"3.9064488",
    "85":"3.9724869",
    "86":"4.0385249",
    "87":"4.104563",
    "88":"4.1706011",
    "89":"4.2366391",
    "90":"4.3026772",
    "91":"4.3687153",
    "92":"4.4347533",
    "93":"4.5007914",
    "94":"4.5668295",
    "95":"4.6328675",
    "96":"4.6989056",
    "97":"4.7649437",
    "98":"4.8309817",
    "99":"4.8970198",
    "100":"4.9630578"
};

var OneYear_Pulse = {
    "49":"0.2145873",
    "59.5":"0.2687747",
    "74.5":"0.5049219",
    "84.5":"0.7361786",
    "94.5":"0.8628367",
    "104.5":"0.8966225",
    "119.5":"0.8578086",
    "139.5":"0.7824602",
    "174.5":"0.6506006",
    "200":"0.5545314"
};

var OneYear_BloodPressure = {
    "79":"0.4071228",
    "89.5":"0.2421043",
    "104.5":"0.0063635",
    "114.5":"-0.1499551",
    "124.5":"-0.2985642",
    "134.5":"-0.4299701",
    "149.5":"-0.5769774",
    "169.5":"-0.6951633",
    "180":"-0.7399799"
};

var OneYear_Creatine =
{
    "0.195":"-0.0999551",
    "0.595":"-0.3049911",
    "0.995":"-0.2920628",
    "1.395":"0.2625441",
    "1.795":"0.4717919",
    "2.995":"0.6085038",
    "4.0":"0.7131714"
};

var OneYear_Killip =
{
    "1": "0",
    "2": "0.63827",
    "3": "0.85325",
    "4": "1.29372"
};

var OneYear_Cardiac =
{
    "no": "0",
    "yes": "0.87185"
};

var OneYear_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0.37660"
};

var OneYear_STEMI =
{
    "no": "0",
    "yes": "0.44303"
};

function CalculateOneYearDeath(age, pulse, bloodPressure, creatinine, killip, cardiac, enzymes, stemi) {
    var ageValue = OneYear_Age[age];
    var pulseValue = OneYear_Pulse[pulse];
    var bloodPressureValue = OneYear_BloodPressure[bloodPressure];
    var killipValue = OneYear_Killip[killip];
    var creatinineValue = OneYear_Creatine[creatinine];
    var cardiacValue = OneYear_Cardiac[cardiac];
    var enzymesValue = OneYear_ElevatedEnzymes[enzymes];
    var stemiValue = OneYear_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9983577131;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
    
    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                   1 Year Death / Substituted                         //
//                                                                      //
//----------------------------------------------------------------------//

var OneYear_Substituted_Age = {
    "29":"1.28267",
    "30":"1.3269",
    "31":"1.37113",
    "32":"1.41536",
    "33":"1.45959",
    "34":"1.50382",
    "35":"1.54805",
    "36":"1.59228",
    "37":"1.63651",
    "38":"1.68074",
    "39":"1.72497",
    "40":"1.7692",
    "41":"1.81343",
    "42":"1.85766",
    "43":"1.90189",
    "44":"1.94612",
    "45":"1.99035",
    "46":"2.03458",
    "47":"2.07881",
    "48":"2.12304",
    "49":"2.1672772",
    "50":"2.2115761",
    "51":"2.2560105",
    "52":"2.3006543",
    "53":"2.3455814",
    "54":"2.3908656",
    "55":"2.4365807",
    "56":"2.4828007",
    "57":"2.5295994",
    "58":"2.5770507",
    "59":"2.6252283",
    "60":"2.6742062",
    "61":"2.7240583",
    "62":"2.7748583",
    "63":"2.8266802",
    "64":"2.8795979",
    "65":"2.9336851",
    "66":"2.9890157",
    "67":"3.0456615",
    "68":"3.1036267",
    "69":"3.1628338",
    "70":"3.2232005",
    "71":"3.2846446",
    "72":"3.3470838",
    "73":"3.4104357",
    "74":"3.4746183",
    "75":"3.539549",
    "76":"3.6051458",
    "77":"3.6713263",
    "78":"3.7380082",
    "79":"3.8051093",
    "80":"3.8725474",
    "81":"3.94024",
    "82":"4.008105",
    "83":"4.0760601",
    "84":"4.1440332",
    "85":"4.2120063",
    "86":"4.2799794",
    "87":"4.3479525",
    "88":"4.4159256",
    "89":"4.4838987",
    "90":"4.5518718",
    "91":"4.6198449",
    "92":"4.687818",
    "93":"4.7557911",
    "94":"4.8237642",
    "95":"4.8917373",
    "96":"4.9597103",
    "97":"5.0276834",
    "98":"5.0956565",
    "99":"5.1636296",
    "100":"5.2316027"
};

var OneYear_Substituted_Pulse = {
    "49":"0.2728483",
    "59.5":"0.340286",
    "74.5":"0.6102857",
    "84.5":"0.8734965",
    "94.5":"1.0277325",
    "104.5":"1.0847059",
    "119.5":"1.0764426",
    "139.5":"1.040713",
    "174.5":"0.9781862",
    "200":"0.9326309"
};

var OneYear_Substituted_BloodPressure = {
    "79":"1.2243367",
    "89.5":"0.9928463",
    "104.5":"0.6621456",
    "114.5":"0.4431929",
    "124.5":"0.23811",
    "134.5":"0.0639759",
    "149.5":"-0.1070916",
    "169.5":"-0.195175",
    "180":"-0.2104199"
};

var OneYear_Substituted_Renal =
{
    "no": "0",
    "yes": "0.44469"
};


var OneYear_Substituted_Diuretic =
{
    "no": "0",
    "yes": "0.68008"
};

var OneYear_Substituted_Cardiac =
{
    "no": "0",
    "yes": "1.18193"
};

var OneYear_Substituted_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0.39944"
};

var OneYear_Substituted_STEMI =
{
    "no": "0",
    "yes": "0.46524"
};


function CalculateOneYearDeathSubstituted(age, pulse, bloodPressure, renal, diuretic, cardiac, enzymes, stemi) {
    //alert("A:" + age + "," + pulse + "," + bloodPressure + "," + renal + "," + diuretic + "," + cardiac + "," + enzymes + "," + stemi);
    var ageValue = OneYear_Substituted_Age[age];
    var pulseValue = OneYear_Substituted_Pulse[pulse];
    var bloodPressureValue = OneYear_Substituted_BloodPressure[bloodPressure];
    var creatinineValue = OneYear_Substituted_Renal[renal];
    var killipValue = OneYear_Substituted_Diuretic[diuretic];
    var cardiacValue = OneYear_Substituted_Cardiac[cardiac];
    var enzymesValue = OneYear_Substituted_ElevatedEnzymes[enzymes];
    var stemiValue = OneYear_Substituted_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9994190637;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
   
    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                        1 Year Death / MI                             //
//                                                                      //
//----------------------------------------------------------------------//

var OneYear_MI_Age = {
    "29":"0.425633",
    "30":"0.44031",
    "31":"0.454987",
    "32":"0.469664",
    "33":"0.484341",
    "34":"0.499018",
    "35":"0.513695",
    "36":"0.528372",
    "37":"0.543049",
    "38":"0.557726",
    "39":"0.572403",
    "40":"0.58708",
    "41":"0.601757",
    "42":"0.616434",
    "43":"0.631111",
    "44":"0.645788",
    "45":"0.660465",
    "46":"0.675142",
    "47":"0.689819",
    "48":"0.704496",
    "49":"0.7191852",
    "50":"0.7339791",
    "51":"0.7490032",
    "52":"0.7643828",
    "53":"0.7802433",
    "54":"0.7967101",
    "55":"0.8139086",
    "56":"0.8319643",
    "57":"0.8510024",
    "58":"0.8711484",
    "59":"0.8925278",
    "60":"0.9152658",
    "61":"0.939488",
    "62":"0.9653197",
    "63":"0.9928863",
    "64":"1.0223132",
    "65":"1.0537258",
    "66":"1.0872495",
    "67":"1.1230062",
    "68":"1.1610029",
    "69":"1.2011081",
    "70":"1.2431822",
    "71":"1.2870853",
    "72":"1.332678",
    "73":"1.3798204",
    "74":"1.4283729",
    "75":"1.4781958",
    "76":"1.5291494",
    "77":"1.5810941",
    "78":"1.6338902",
    "79":"1.6873979",
    "80":"1.7414776",
    "81":"1.7959897",
    "82":"1.8507944",
    "83":"1.905752",
    "84":"1.9607402",
    "85":"2.0157283",
    "86":"2.0707165",
    "87":"2.1257047",
    "88":"2.1806929",
    "89":"2.2356811",
    "90":"2.2906693",
    "91":"2.3456574",
    "92":"2.4006456",
    "93":"2.4556338",
    "94":"2.510622",
    "95":"2.5656102",
    "96":"2.6205984",
    "97":"2.6755865",
    "98":"2.7305747",
    "99":"2.7855629",
    "100":"2.8405511"
};

var OneYear_MI_Pulse = {
    "49":"0.06909",
    "59.5":"0.0916176",
    "74.5":"0.2734092",
    "84.5":"0.4685902",
    "94.5":"0.5750222",
    "104.5":"0.6017448",
    "119.5":"0.5650198",
    "139.5":"0.4958011",
    "174.5":"0.3746683",
    "200":"0.2864144"
};

var OneYear_MI_BloodPressure = {
    "79":"0.6609103",
    "89.5":"0.529623",
    "104.5":"0.3420698",
    "114.5":"0.2178761",
    "124.5":"0.101392",
    "134.5":"0.0021112",
    "149.5":"-0.0967086",
    "169.5":"-0.1506446",
    "180":"-0.1617299"
};

var OneYear_MI_Creatine =
{
    "0.195":"-0.0311318",
    "0.595":"-0.0949918",
    "0.995":"-0.0134909",
    "1.395":"0.4146596",
    "1.795":"0.5860543",
    "2.995":"0.7379644",
    "4.0":"0.8579377"
};

var OneYear_MI_Killip =
{
    "1": "0",
    "2": "0.53625",
    "3": "0.68594",
    "4": "1.15850"
};

var OneYear_MI_Cardiac =
{
    "no": "0",
    "yes": "0.67071"
};

var OneYear_MI_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0.22710"
};

var OneYear_MI_STEMI =
{
    "no": "0",
    "yes": "0.32831"
};

function CalculateOneYearDeathMI(age, pulse, bloodPressure, creatinine, killip, cardiac, enzymes, stemi) {
    var ageValue = OneYear_MI_Age[age];
    var pulseValue = OneYear_MI_Pulse[pulse];
    var bloodPressureValue = OneYear_MI_BloodPressure[bloodPressure];
    var killipValue = OneYear_MI_Killip[killip];
    var creatinineValue = OneYear_MI_Creatine[creatinine];
    var cardiacValue = OneYear_MI_Cardiac[cardiac];
    var enzymesValue = OneYear_MI_ElevatedEnzymes[enzymes];
    var stemiValue = OneYear_MI_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9865696068;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
    
    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                 1 Year Death / Substituted / MI                      //
//                                                                      //
//----------------------------------------------------------------------//

var OneYear_Substituted_MI_Age = {
    "29":"0.513068",
    "30":"0.53076",
    "31":"0.548452",
    "32":"0.566144",
    "33":"0.583836",
    "34":"0.601528",
    "35":"0.61922",
    "36":"0.636912",
    "37":"0.654604",
    "38":"0.672296",
    "39":"0.689988",
    "40":"0.70768",
    "41":"0.725372",
    "42":"0.743064",
    "43":"0.760756",
    "44":"0.778448",
    "45":"0.79614",
    "46":"0.813832",
    "47":"0.831524",
    "48":"0.849216",
    "49":"0.8669197",
    "50":"0.8847244",
    "51":"0.9027507",
    "52":"0.9211194",
    "53":"0.9399513",
    "54":"0.9593671",
    "55":"0.9794878",
    "56":"1.0004339",
    "57":"1.0223264",
    "58":"1.0452859",
    "59":"1.0694333",
    "60":"1.0948894",
    "61":"1.1217749",
    "62":"1.1502106",
    "63":"1.1803173",
    "64":"1.2122158",
    "65":"1.2460269",
    "66":"1.2818712",
    "67":"1.3198663",
    "68":"1.3600189",
    "69":"1.4022023",
    "70":"1.4462819",
    "71":"1.4921233",
    "72":"1.539592",
    "73":"1.5885533",
    "74":"1.6388727",
    "75":"1.6904158",
    "76":"1.7430479",
    "77":"1.7966346",
    "78":"1.8510413",
    "79":"1.9061334",
    "80":"1.9617764",
    "81":"2.0178359",
    "82":"2.0741772",
    "83":"2.1306658",
    "84":"2.1871838",
    "85":"2.2437019",
    "86":"2.3002199",
    "87":"2.3567379",
    "88":"2.413256",
    "89":"2.469774",
    "90":"2.526292",
    "91":"2.5828101",
    "92":"2.6393281",
    "93":"2.6958461",
    "94":"2.7523642",
    "95":"2.8088822",
    "96":"2.8654002",
    "97":"2.9219183",
    "98":"2.9784363",
    "99":"3.0349543",
    "100":"3.0914724"
};

var OneYear_Substituted_MI_Pulse = {
    "49":"0.0504863",
    "59.5":"0.0699143",
    "74.5":"0.2646505",
    "84.5":"0.4814061",
    "94.5":"0.608825",
    "104.5":"0.655008",
    "119.5":"0.6460028",
    "139.5":"0.6133558",
    "174.5":"0.5562236",
    "200":"0.5145987"
};

var OneYear_Substituted_MI_BloodPressure = 
{
    "79":"1.2166452",
    "89.5":"1.0395593",
    "104.5":"0.7865796",
    "114.5":"0.619228",
    "124.5":"0.4637959",
    "134.5":"0.3349614",
    "149.5":"0.2191705",
    "169.5":"0.1851043",
    "180":"0.1938601"
};

var OneYear_Substituted_MI_Renal =
{
    "no": "0",
    "yes": "0.47766"
};


var OneYear_Substituted_MI_Diuretic =
{
    "no": "0",
    "yes": "0.53363"
};

var OneYear_Substituted_MI_Cardiac =
{
    "no": "0",
    "yes": "0.92949"
};

var OneYear_Substituted_MI_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0.25469"
};

var OneYear_Substituted_MI_STEMI =
{
    "no": "0",
    "yes": "0.34746"
};

function CalculateOneYearDeathSubstitutedMI(age, pulse, bloodPressure, renal, diuretic, cardiac, enzymes, stemi) {
    //alert("A:" + age + "," + pulse + "," + bloodPressure + "," + renal + "," + diuretic + "," + cardiac + "," + enzymes + "," + stemi);
    var ageValue = OneYear_Substituted_MI_Age[age];
    var pulseValue = OneYear_Substituted_MI_Pulse[pulse];
    var bloodPressureValue = OneYear_Substituted_MI_BloodPressure[bloodPressure];
    var creatinineValue = OneYear_Substituted_MI_Renal[renal];
    var killipValue = OneYear_Substituted_MI_Diuretic[diuretic];
    var cardiacValue = OneYear_Substituted_MI_Cardiac[cardiac];
    var enzymesValue = OneYear_Substituted_MI_ElevatedEnzymes[enzymes];
    var stemiValue = OneYear_Substituted_MI_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9919559302;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
   
    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                            3 Year Death                              //
//                                                                      //
//----------------------------------------------------------------------//

var ThreeYear_Age = {
    "29":"1.767115",
    "30":"1.82805",
    "31":"1.888985",
    "32":"1.94992",
    "33":"2.010855",
    "34":"2.07179",
    "35":"2.132725",
    "36":"2.19366",
    "37":"2.254595",
    "38":"2.31553",
    "39":"2.376465",
    "40":"2.4374",
    "41":"2.498335",
    "42":"2.55927",
    "43":"2.620205",
    "44":"2.68114",
    "45":"2.742075",
    "46":"2.80301",
    "47":"2.863945",
    "48":"2.92488",
    "49":"2.985815",
    "50":"3.04675",
    "51":"3.107685",
    "52":"3.16862",
    "53":"3.229555",
    "54":"3.29049",
    "55":"3.351425",
    "56":"3.41236",
    "57":"3.473295",
    "58":"3.53423",
    "59":"3.595165",
    "60":"3.6561",
    "61":"3.717035",
    "62":"3.77797",
    "63":"3.838905",
    "64":"3.89984",
    "65":"3.960775",
    "66":"4.02171",
    "67":"4.082645",
    "68":"4.14358",
    "69":"4.204515",
    "70":"4.26545",
    "71":"4.326385",
    "72":"4.38732",
    "73":"4.448255",
    "74":"4.50919",
    "75":"4.570125",
    "76":"4.63106",
    "77":"4.691995",
    "78":"4.75293",
    "79":"4.813865",
    "80":"4.8748",
    "81":"4.935735",
    "82":"4.99667",
    "83":"5.057605",
    "84":"5.11854",
    "85":"5.179475",
    "86":"5.24041",
    "87":"5.301345",
    "88":"5.36228",
    "89":"5.423215",
    "90":"5.48415",
    "91":"5.545085",
    "92":"5.60602",
    "93":"5.666955",
    "94":"5.72789",
    "95":"5.788825",
    "96":"5.84976",
    "97":"5.910695",
    "98":"5.97163",
    "99":"6.032565",
    "100":"6.0935"
};

var ThreeYear_Pulse = {
    "49":"-0.0889187",
    "59.5":"-0.0910469",
    "74.5":"0.2343003",
    "84.5":"0.6246916",
    "94.5":"0.8450473",
    "104.5":"0.9101415",
    "119.5":"0.8581709",
    "139.5":"0.7494289",
    "174.5":"0.5591304",
    "200":"0.4204844"
};

var ThreeYear_BloodPressure = {
    "79":"3.5872931",
    "89.5":"3.0903948",
    "104.5":"2.6394952",
    "114.5":"2.442021",
    "124.5":"2.2607121",
    "134.5":"2.0407927",
    "149.5":"1.6991854",
    "169.5":"1.5533218",
    "180":"1.5973204"
};

var ThreeYear_Creatine =
{
    "0.195":"0.0768359",
    "0.595":"0.2344479",
    "0.995":"0.3920599",
    "1.395":"0.5496719",
    "1.795":"0.7072839",
    "2.995":"1.1801199",
    "4.0":"1.57612"
};

var ThreeYear_Killip =
{
    "1": "0",
    "2": "0.08466",
    "3": "0.82842",
    "4": "0.82842"
};

var ThreeYear_Cardiac =
{
    "no": "0",
    "yes": "1.07623"
};

var ThreeYear_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0"
};

var ThreeYear_STEMI =
{
    "no": "0",
    "yes": "0.41228"
};

function CalculateThreeYearDeath(age, pulse, bloodPressure, creatinine, killip, cardiac, enzymes, stemi) {
    var ageValue = ThreeYear_Age[age];
    var pulseValue = ThreeYear_Pulse[pulse];
    var bloodPressureValue = ThreeYear_BloodPressure[bloodPressure];
    var killipValue = ThreeYear_Killip[killip];
    var creatinineValue = ThreeYear_Creatine[creatinine];
    var cardiacValue = ThreeYear_Cardiac[cardiac];
    var enzymesValue = ThreeYear_ElevatedEnzymes[enzymes];
    var stemiValue = ThreeYear_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9998715509;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
    //result = result * (93 / 120);
   
    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                      3 Year Death / Substituted                      //
//                                                                      //
//----------------------------------------------------------------------//

var ThreeYear_Substituted_Age = {
    "29":"1.710942",
    "30":"1.76994",
    "31":"1.828938",
    "32":"1.887936",
    "33":"1.946934",
    "34":"2.005932",
    "35":"2.06493",
    "36":"2.123928",
    "37":"2.182926",
    "38":"2.241924",
    "39":"2.300922",
    "40":"2.35992",
    "41":"2.418918",
    "42":"2.477916",
    "43":"2.536914",
    "44":"2.595912",
    "45":"2.65491",
    "46":"2.713908",
    "47":"2.772906",
    "48":"2.831904",
    "49":"2.890902",
    "50":"2.9499",
    "51":"3.008898",
    "52":"3.067896",
    "53":"3.126894",
    "54":"3.185892",
    "55":"3.24489",
    "56":"3.303888",
    "57":"3.362886",
    "58":"3.421884",
    "59":"3.480882",
    "60":"3.53988",
    "61":"3.598878",
    "62":"3.657876",
    "63":"3.716874",
    "64":"3.775872",
    "65":"3.83487",
    "66":"3.893868",
    "67":"3.952866",
    "68":"4.011864",
    "69":"4.070862",
    "70":"4.12986",
    "71":"4.188858",
    "72":"4.247856",
    "73":"4.306854",
    "74":"4.365852",
    "75":"4.42485",
    "76":"4.483848",
    "77":"4.542846",
    "78":"4.601844",
    "79":"4.660842",
    "80":"4.71984",
    "81":"4.778838",
    "82":"4.837836",
    "83":"4.896834",
    "84":"4.955832",
    "85":"5.01483",
    "86":"5.073828",
    "87":"5.132826",
    "88":"5.191824",
    "89":"5.250822",
    "90":"5.30982",
    "91":"5.368818",
    "92":"5.427816",
    "93":"5.486814",
    "94":"5.545812",
    "95":"5.60481",
    "96":"5.663808",
    "97":"5.722806",
    "98":"5.781804",
    "99":"5.840802",
    "100":"5.8998"
};

var ThreeYear_Substituted_Pulse = {
    "49":"-0.1549543",
    "59.5":"-0.1704667",
    "74.5":"0.1507527",
    "84.5":"0.5491089",
    "94.5":"0.7759076",
    "104.5":"0.8453205",
    "119.5":"0.7977809",
    "139.5":"0.6944071",
    "174.5":"0.5135029",
    "200":"0.3817013"
};

var ThreeYear_Substituted_BloodPressure = {
    "79":"4.5695482",
    "89.5":"3.9853504",
    "104.5":"3.4167333",
    "114.5":"3.1524173",
    "124.5":"2.9277477",
    "134.5":"2.7001331",
    "149.5":"2.4021178",
    "169.5":"2.3409886",
    "180":"2.4324792"
};

var ThreeYear_Substituted_Renal =
{
    "no": "0",
    "yes": "0.69806"
};

var ThreeYear_Substituted_Diuretic =
{
    "no": "0",
    "yes": "0.68142"
};

var ThreeYear_Substituted_Cardiac =
{
    "no": "0",
    "yes": "1.17867"
};

var ThreeYear_Substituted_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0"
};

var ThreeYear_Substituted_STEMI =
{
    "no": "0",
    "yes": "0.32912"
};

function CalculateThreeYearDeathSubstituted(age, pulse, bloodPressure, renal, diuretic, cardiac, enzymes, stemi) {
    //alert("A:" + age + "," + pulse + "," + bloodPressure + "," + renal + "," + diuretic + "," + cardiac + "," + enzymes + "," + stemi);
    var ageValue = ThreeYear_Substituted_Age[age];
    var pulseValue = ThreeYear_Substituted_Pulse[pulse];
    var bloodPressureValue = ThreeYear_Substituted_BloodPressure[bloodPressure];
    var creatinineValue = ThreeYear_Substituted_Renal[renal];
    var killipValue = ThreeYear_Substituted_Diuretic[diuretic];
    var cardiacValue = ThreeYear_Substituted_Cardiac[cardiac];
    var enzymesValue = ThreeYear_Substituted_ElevatedEnzymes[enzymes];
    var stemiValue = ThreeYear_Substituted_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9998961080;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
    //result = result * (93 / 120);
   
    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                        3 Year Death / MI                             //
//                                                                      //
//----------------------------------------------------------------------//

var ThreeYear_MI_Age = {
    "29":"1.255671",
    "30":"1.29897",
    "31":"1.342269",
    "32":"1.385568",
    "33":"1.428867",
    "34":"1.472166",
    "35":"1.515465",
    "36":"1.558764",
    "37":"1.602063",
    "38":"1.645362",
    "39":"1.688661",
    "40":"1.73196",
    "41":"1.775259",
    "42":"1.818558",
    "43":"1.861857",
    "44":"1.905156",
    "45":"1.948455",
    "46":"1.991754",
    "47":"2.035053",
    "48":"2.078352",
    "49":"2.121651",
    "50":"2.16495",
    "51":"2.208249",
    "52":"2.251548",
    "53":"2.294847",
    "54":"2.338146",
    "55":"2.381445",
    "56":"2.424744",
    "57":"2.468043",
    "58":"2.511342",
    "59":"2.554641",
    "60":"2.59794",
    "61":"2.641239",
    "62":"2.684538",
    "63":"2.727837",
    "64":"2.771136",
    "65":"2.814435",
    "66":"2.857734",
    "67":"2.901033",
    "68":"2.944332",
    "69":"2.987631",
    "70":"3.03093",
    "71":"3.074229",
    "72":"3.117528",
    "73":"3.160827",
    "74":"3.204126",
    "75":"3.247425",
    "76":"3.290724",
    "77":"3.334023",
    "78":"3.377322",
    "79":"3.420621",
    "80":"3.46392",
    "81":"3.507219",
    "82":"3.550518",
    "83":"3.593817",
    "84":"3.637116",
    "85":"3.680415",
    "86":"3.723714",
    "87":"3.767013",
    "88":"3.810312",
    "89":"3.853611",
    "90":"3.89691",
    "91":"3.940209",
    "92":"3.983508",
    "93":"4.026807",
    "94":"4.070106",
    "95":"4.113405",
    "96":"4.156704",
    "97":"4.200003",
    "98":"4.243302",
    "99":"4.286601",
    "100":"4.3299"
};

var ThreeYear_MI_Pulse = {
    "49":"0.2955843",
    "59.5":"0.3589238",
    "74.5":"0.4494088",
    "84.5":"0.5097322",
    "94.5":"0.5700555",
    "104.5":"0.6303788",
    "119.5":"0.7208638",
    "139.5":"0.8415105",
    "174.5":"1.0526422",
    "200":"1.2064667"
};

var ThreeYear_MI_BloodPressure = {
    "79":"-0.5735795",
    "89.5":"-0.6498148",
    "104.5":"-0.7587223",
    "114.5":"-0.8313273",
    "124.5":"-0.9039323",
    "134.5":"-0.9765373",
    "149.5":"-1.0854448",
    "169.5":"-1.2306548",
    "180":"-1.30689"
};

var ThreeYear_MI_Creatine =
{
    "0.195":".0700928",
    "0.595":".2138728",
    "0.995":".3576528",
    "1.395":".5014328",
    "1.795":".6452128",
    "2.995":"1.0765528",
    "4.0":    "1.4378"
};

var ThreeYear_MI_Killip =
{
    "1": "0",
    "2": "0.21991",
    "3": "1.06610",
    "4": "1.06610"
};

var ThreeYear_MI_Cardiac =
{
    "no": "0",
    "yes": "0.61622"
};

var ThreeYear_MI_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0"
};

var ThreeYear_MI_STEMI =
{
    "no": "0",
    "yes": "0.42540"
};

function CalculateThreeYearDeathMI(age, pulse, bloodPressure, creatinine, killip, cardiac, enzymes, stemi) {
    var ageValue = ThreeYear_MI_Age[age];
    var pulseValue = ThreeYear_MI_Pulse[pulse];
    var bloodPressureValue = ThreeYear_MI_BloodPressure[bloodPressure];
    var killipValue = ThreeYear_MI_Killip[killip];
    var creatinineValue = ThreeYear_MI_Creatine[creatinine];
    var cardiacValue = ThreeYear_MI_Cardiac[cardiac];
    var enzymesValue = ThreeYear_MI_ElevatedEnzymes[enzymes];
    var stemiValue = ThreeYear_MI_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9875014749;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
    //result = result * (14 / 18);

    return result.roundup(1);
}


//----------------------------------------------------------------------//
//                                                                      //
//                 3 Year Death / Substituted / MI                      //
//                                                                      //
//----------------------------------------------------------------------//

var ThreeYear_Substituted_MI_Age = {
    "29":"1.223742",
    "30":"1.26594",
    "31":"1.308138",
    "32":"1.350336",
    "33":"1.392534",
    "34":"1.434732",
    "35":"1.47693",
    "36":"1.519128",
    "37":"1.561326",
    "38":"1.603524",
    "39":"1.645722",
    "40":"1.68792",
    "41":"1.730118",
    "42":"1.772316",
    "43":"1.814514",
    "44":"1.856712",
    "45":"1.89891",
    "46":"1.941108",
    "47":"1.983306",
    "48":"2.025504",
    "49":"2.067702",
    "50":"2.1099",
    "51":"2.152098",
    "52":"2.194296",
    "53":"2.236494",
    "54":"2.278692",
    "55":"2.32089",
    "56":"2.363088",
    "57":"2.405286",
    "58":"2.447484",
    "59":"2.489682",
    "60":"2.53188",
    "61":"2.574078",
    "62":"2.616276",
    "63":"2.658474",
    "64":"2.700672",
    "65":"2.74287",
    "66":"2.785068",
    "67":"2.827266",
    "68":"2.869464",
    "69":"2.911662",
    "70":"2.95386",
    "71":"2.996058",
    "72":"3.038256",
    "73":"3.080454",
    "74":"3.122652",
    "75":"3.16485",
    "76":"3.207048",
    "77":"3.249246",
    "78":"3.291444",
    "79":"3.333642",
    "80":"3.37584",
    "81":"3.418038",
    "82":"3.460236",
    "83":"3.502434",
    "84":"3.544632",
    "85":"3.58683",
    "86":"3.629028",
    "87":"3.671226",
    "88":"3.713424",
    "89":"3.755622",
    "90":"3.79782",
    "91":"3.840018",
    "92":"3.882216",
    "93":"3.924414",
    "94":"3.966612",
    "95":"4.00881",
    "96":"4.051008",
    "97":"4.093206",
    "98":"4.135404",
    "99":"4.177602",
    "100":"4.2198"
};

var ThreeYear_Substituted_MI_Pulse = {
    "49":"0.3671733",
    "59.5":"0.4458533",
    "74.5":"0.5582533",
    "84.5":"0.6331867",
    "94.5":"0.70812",
    "104.5":"0.7830533",
    "119.5":"0.8954533",
    "139.5":"1.04532",
    "174.5":"1.3075867",
    "200":"1.4986667"
};

var ThreeYear_Substituted_MI_BloodPressure = {
    "79":"1.105618",
    "89.5":".9511564",
    "104.5":"0.7304969",
    "114.5":"0.5845485",
    "124.5":"0.4492046",
    "134.5":"0.3375235",
    "149.5":"0.238916",
    "169.5":"0.2144852",
    "180":"0.2253601"
};

var ThreeYear_Substituted_MI_Renal =
{
    "no": "0",
    "yes": "0.51167"
};

var ThreeYear_Substituted_MI_Diuretic =
{
    "no": "0",
    "yes": "0.65284"
};

var ThreeYear_Substituted_MI_Cardiac =
{
    "no": "0",
    "yes": "0.72786"
};

var ThreeYear_Substituted_MI_ElevatedEnzymes =
{
    "no": "0",
    "yes": "0"
};

var ThreeYear_Substituted_MI_STEMI =
{
    "no": "0",
    "yes": "0.38290"
};

function CalculateThreeYearDeathSubstitutedMI(age, pulse, bloodPressure, renal, diuretic, cardiac, enzymes, stemi) {
    //alert("A:" + age + "," + pulse + "," + bloodPressure + "," + renal + "," + diuretic + "," + cardiac + "," + enzymes + "," + stemi);
    var ageValue = ThreeYear_Substituted_MI_Age[age];
    var pulseValue = ThreeYear_Substituted_MI_Pulse[pulse];
    var bloodPressureValue = ThreeYear_Substituted_MI_BloodPressure[bloodPressure];
    var creatinineValue = ThreeYear_Substituted_MI_Renal[renal];
    var killipValue = ThreeYear_Substituted_MI_Diuretic[diuretic];
    var cardiacValue = ThreeYear_Substituted_MI_Cardiac[cardiac];
    var enzymesValue = ThreeYear_Substituted_MI_ElevatedEnzymes[enzymes];
    var stemiValue = ThreeYear_Substituted_MI_STEMI[stemi];
    //alert(ageValue + "," + pulseValue + "," + bloodPressureValue + "," + cardiacValue + "," + enzymesValue + "," + stemiValue + "," + killipValue);
    var xbhat = parseFloat(ageValue) + parseFloat(pulseValue) + parseFloat(bloodPressureValue) + parseFloat(creatinineValue) + parseFloat(cardiacValue) + parseFloat(enzymesValue) + parseFloat(stemiValue) + parseFloat(killipValue);
    var sZeroT = 0.9958878333;
    var temp = Math.exp(xbhat);
    var temp2 = Math.pow(sZeroT, temp);
    var result = 1 - temp2;
    result = result * 100;
    //result = result * (14 / 18);
    
    return result.roundup(1);
}