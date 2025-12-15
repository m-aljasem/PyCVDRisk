var testMode = false;
var reviewMode = false;
var startAnimation;
var cancelTimeout = false;
var isAndroid = navigator.userAgent.toLowerCase().indexOf("android") > -1;


function GetQueryStringParams(sParam) {
    var sPageURL = window.location.search.substring(1);
    var sURLVariables = sPageURL.split('&');
    for (var i = 0; i < sURLVariables.length; i++) {
        var sParameterName = sURLVariables[i].split('=');
        if (sParameterName[0] == sParam) {
            return sParameterName[1];
        }
    }
}

if (GetQueryStringParams('reviewMode') == 'rpg') {
    reviewMode = true;
    $('#reviewMode').show();
    $('#inHospitalMiContainer').show();
    $('#sixMonthMiContainer').show();
}



//////////////////////  Functions to add/remove classes from page containers placed in queue for page transition effects
function clearHideClass() {
    $(this).removeClass('hide-left hide-right').dequeue();
}

function clearActiveClass() {
    $(this).removeClass('active').dequeue();
}

function clearAllClass() {
    $(this).removeClass('active hide-left hide-right').dequeue();
}

function addActiveClass() {
    $(this).addClass('active').dequeue();
}


function log10(val) {
    return Math.log(val) / Math.log(10);
}







//------------------------------------------------------------------------//
//                                                                        //   
//                    Calculations - Do Not Modify                        //
//                                                                        // 
//------------------------------------------------------------------------//

var resultInHospital;
var resultInHospitalMIArray;
var resultInHospitalMI;
var resultSixMonthArray;
var resultSixMonth;
var resultSixMonthUnadjusted;

var resultSixMonthMIArray;
var resultSixMonthMI;
var resultSixMonthMIUnadjusted;
var resultOneYear;
var resultOneYearMI;
var resultThreeYear;
var resultThreeYearMI;

var resultInHospitalToPlotOnGraph ;
var resultSixMonthToPlotOnGraph;
var resultOneYearToPlotOnGraph;
var resultOneYearMIToPlotOnGraph;
var resultThreeYearToPlotOnGraph;
var resultThreeYearMIToPlotOnGraph;

function calculateResults() {

    resultInHospital = CalculateAdmissionToInHospitalDeath(input_age, input_heartRate, input_bloodPressure, input_creatine, input_renal, input_chf, input_diuretic, input_cardiac, input_troponin, input_segment);
    resultInHospitalMIArray = CalculateAdmissionToInHospitalDeathMI(input_age, input_heartRate, input_bloodPressure, input_creatine, input_renal, input_chf, input_diuretic, input_cardiac, input_troponin, input_segment);
    resultInHospitalMI = resultInHospitalMIArray[0];
    resultSixMonthArray = CalculateSixMonthDeath(input_age, input_heartRate, input_bloodPressure, input_creatine, input_renal, input_chf, input_diuretic, input_cardiac, input_troponin, input_segment);
    resultSixMonth = resultSixMonthArray[0];
    resultSixMonthUnadjusted = resultSixMonthArray[2];

    resultSixMonthMIArray = CalculateSixMonthDeathMI(input_age, input_heartRate, input_bloodPressure, input_creatine, input_renal, input_chf, input_diuretic, input_cardiac, input_troponin, input_segment);
    resultSixMonthMI = resultSixMonthMIArray[0];
    resultSixMonthMIUnadjusted = resultSixMonthMIArray[2];

    var substituted = false;

    if (input_chf == "na" && input_creatine != "na") {
        substituted = true;
        if (parseFloat(input_creatine) > 1.59) {
            input_renal = "yes";
        }
        else {
            input_renal = "no";
        }
    }
    else if (input_creatine == "na" && input_chf != "na") {
        substituted = true;
        if (input_chf == "3" || input_chf == "4") {
            input_diuretic = "yes";
        }
        else {
            input_diuretic = "no";
        }
    }
    else if (input_creatine == "na" && input_chf == "na") {
        substituted = true;
    }

    if (substituted) {
        resultOneYear = CalculateOneYearDeathSubstituted(input_age, input_heartRate, input_bloodPressure, input_renal, input_diuretic, input_cardiac, input_troponin, input_segment);
        resultOneYearMI = CalculateOneYearDeathSubstitutedMI(input_age, input_heartRate, input_bloodPressure, input_renal, input_diuretic, input_cardiac, input_troponin, input_segment);
        resultThreeYear = CalculateThreeYearDeathSubstituted(input_age, input_heartRate, input_bloodPressure, input_renal, input_diuretic, input_cardiac, input_troponin, input_segment);
        resultThreeYearMI = CalculateThreeYearDeathSubstitutedMI(input_age, input_heartRate, input_bloodPressure, input_renal, input_diuretic, input_cardiac, input_troponin, input_segment);
    }
    else {
        resultOneYear = CalculateOneYearDeath(input_age, input_heartRate, input_bloodPressure, input_creatine, input_chf, input_cardiac, input_troponin, input_segment);
        resultOneYearMI = CalculateOneYearDeathMI(input_age, input_heartRate, input_bloodPressure, input_creatine, input_chf, input_cardiac, input_troponin, input_segment);
        resultThreeYear = CalculateThreeYearDeath(input_age, input_heartRate, input_bloodPressure, input_creatine, input_chf, input_cardiac, input_troponin, input_segment);
        resultThreeYearMI = CalculateThreeYearDeathMI(input_age, input_heartRate, input_bloodPressure, input_creatine, input_chf, input_cardiac, input_troponin, input_segment);
    }

    if (resultOneYear < resultSixMonth) {
        var midResult = parseFloat(resultOneYear) + ((parseFloat(resultSixMonth) - parseFloat(resultOneYear)) / 2);
        midResult = midResult.roundup(1);
        if (resultOneYear.toString().length == 1) {
            resultOneYear = resultOneYear.toFixed(1);
        }
        if (resultSixMonth.toString().length == 1) {
            resultSixMonth = resultSixMonth.toFixed(1);
        }
        $('#sixMonthResult').html(resultOneYear + "&ndash;" + resultSixMonth + " <span class=\"addbr-result\"></span>(" + resultSixMonthArray[1].roundup(1) + ")");
        $('#oneYearResult').html(resultOneYear + "&ndash;" + resultSixMonth);
        if (!reviewMode) {
            resultOneYear = midResult;
            resultSixMonth = midResult;
        }
    }
    else {
        if (resultSixMonth > 91) {
            resultSixMonth = 91;
            $('#sixMonthResult').html(">" + resultSixMonth + " <span class=\"addbr-result\"></span>(" + resultSixMonthArray[1].roundup(1) + ")");
        }
        else {
            if (resultSixMonth.toString().length == 1) {
                resultSixMonth = resultSixMonth.toFixed(1);
            }
            $('#sixMonthResult').html(resultSixMonth + " <span class=\"addbr-result\"></span>(" + resultSixMonthArray[1].roundup(1) + ")");
        }
        if (resultOneYear > 91) {
            resultOneYear = 91;
            $('#oneYearResult').html(">" + resultOneYear);
        }
        else {
            if (resultOneYear.toString().length == 1) {
                resultOneYear = resultOneYear.toFixed(1);
            }
            $('#oneYearResult').html(resultOneYear);
        }
    }

    if (resultInHospital > 91) {
        resultInHospital = 91;
        $('#inHospitalResult').html(">" + resultInHospital);
    }
    else {
        if (resultInHospital.toString().length == 1) {
            resultInHospital = resultInHospital.toFixed(1);
        }
        $('#inHospitalResult').html(resultInHospital);
    }

    if (resultInHospitalMI.toString().length == 1) {
        resultInHospitalMI = resultInHospitalMI.toFixed(1);
    }
    if (resultSixMonthMI.toString().length == 1) {
        resultSixMonthMI = resultSixMonthMI.toFixed(1);
    }

    $('#inHospitalResultMI').html(resultInHospitalMI);
    $('#sixMonthResultMI').html(resultSixMonthMI);

    if (testMode) {
        $('#sixMonthResult').addClass('test-score').html(resultSixMonth + "<br />(Score:" + resultSixMonthArray[1] + ")");
    }
    else {
        if ($('#sixMonthResult').hasClass('test-score')) {
            $(this).removeClass('test-score');
        }

    }

    if (parseFloat(resultOneYear) >= parseFloat(resultOneYearMI)) {
        resultOneYearMI = resultOneYear;
        $('#oneYearResultMI').html('>' + resultOneYear);
    }
    else {
        if (resultOneYearMI > 91) {
            resultOneYearMI = 91;
            $('#oneYearResultMI').html(">" + resultOneYearMI);
        }
        else {
            if (resultOneYearMI.toString().length == 1) {
                resultOneYearMI = resultOneYearMI.toFixed(1);
            }
            $('#oneYearResultMI').html(resultOneYearMI);
        }
    }

    if (resultThreeYear > 91) {
        resultThreeYear = 91;
        $('#threeYearResult').html(">" + resultThreeYear);
    }
    else {
        if (resultThreeYear.toString().length == 1) {
            resultThreeYear = resultThreeYear.toFixed(1);
        }
        $('#threeYearResult').html(resultThreeYear);
    }

    if (parseFloat(resultThreeYear) >= parseFloat(resultThreeYearMI)) {
        resultThreeYearMI = resultThreeYear;
        $('#threeYearResultMI').html('>' + resultThreeYear);
    }
    else {
        if (resultThreeYearMI > 91) {
            resultThreeYearMI = 91;
            $('#threeYearResultMI').html(">" + resultThreeYearMI);
        }
        else {
            if (resultThreeYearMI.toString().length == 1) {
                resultThreeYearMI = resultThreeYearMI.toFixed(1);
            }
            $('#threeYearResultMI').html(resultThreeYearMI);
        }
    }

    if (reviewMode) {
        $('#inHospitalResult').html($('#inHospitalResult').html() + ", Unadjusted result: " + resultInHospital);
        $('#sixMonthResult').html($('#sixMonthResult').html() + ", Unadjusted result: " + resultSixMonthUnadjusted);
        $('#oneYearResult').html($('#oneYearResult').html() + ", Unadjusted result: " + resultOneYear);
        $('#threeYearResult').html($('#threeYearResult').html() + ", Unadjusted result: " + resultThreeYear);
        $('#inHospitalResultMI').html($('#inHospitalResultMI').html() + ", Unadjusted result: " + resultInHospitalMI);
        $('#sixMonthResultMI').html($('#sixMonthResultMI').html() + ", Unadjusted result: " + resultSixMonthMIUnadjusted);
        $('#oneYearResultMI').html($('#oneYearResultMI').html() + ", Unadjusted result: " + resultOneYearMI);
        $('#threeYearResultMI').html($('#threeYearResultMI').html() + ", Unadjusted result: " + resultThreeYearMI);
    }

    //EXAMPLES OF HOW TO GET WHAT POINT TO PLOT
    resultInHospitalToPlotOnGraph = log10(resultInHospital) * 50;
    //var resultInHospitalMIToPlotOnGraph = log10(resultInHospitalMI) * 50;
    resultSixMonthToPlotOnGraph = log10(resultSixMonth) * 50;
    //var resultSixMonthMIToPlotOnGraph = log10(resultSixMonthMI) * 50;
    resultOneYearToPlotOnGraph = log10(resultOneYear) * 50;
    resultOneYearMIToPlotOnGraph = log10(resultOneYearMI) * 50;
    resultThreeYearToPlotOnGraph = log10(resultThreeYear) * 50;
    resultThreeYearMIToPlotOnGraph = log10(resultThreeYearMI) * 50;
    //alert(resultInHospitalToPlotOnGraph + "," + resultInHospitalMIToPlotOnGraph + "," + resultSixMonthToPlotOnGraph + "," + resultSixMonthMIToPlotOnGraph + "," + resultOneYearToPlotOnGraph + "," + resultOneYearMIToPlotOnGraph + "," + resultThreeYearToPlotOnGraph + "," + resultThreeYearMIToPlotOnGraph);




}



