/**
 *
 */
function getTargetValues(risk) {
	switch(risk) {
		case "0":
			return [160, 190];
		case "1":
			return [140, 170];
		case "2":
			return [120, 150];
	}
}

/**
 *
 */
function updateView() {
	var param = getCGIParam();
	
	let target_vals = getTargetValues(param.risk);
	
	document.getElementById("rateText").innerHTML = param.rateText;
	document.getElementById("riskText").innerHTML = param.riskText;
	document.getElementById("targetLDLCText").innerHTML = target_vals[0];
	document.getElementById("targetNonLDLCText").innerHTML = target_vals[1];

	var rate_div = floatFormat(param.rate / param.rateMin, 1);
	if(rate_div <= 1.0) {
		document.getElementById("highRiskMsg").innerHTML = "";
	} else {
		document.getElementById("lowRiskMsg").innerHTML = "";
		document.getElementById("rateDiv").innerHTML = rate_div;
	}

	document.getElementById("old").innerHTML = param.old;
	document.getElementById("sex").innerHTML = param.sex;
	document.getElementById("smoking").innerHTML = param.smoking;
	document.getElementById("ldl").innerHTML = param.ldl;
	document.getElementById("hdl").innerHTML = param.hdl;
	document.getElementById("systolicBloodPressure").innerHTML = param.systolicBloodPressure;
	document.getElementById("diabeteMellitus").innerHTML = param.diabeteMellitus;
	
	if(param.risk >= 2) {
		document.getElementById("lowRiskOnlyMsg").innerHTML = "";
	}
}

function floatFormat( number, n )
{
	var _pow = Math.pow( 10 , n ) ;
	return Math.round( number * _pow ) / _pow ;
}

$(function() {
	//
	updateView();
	//
	$("#editBtn").click(function() {
		history.back();
	});	
	//
	$("#backBtn").click(function() {
		window.location.href = "index.html";
	});
});
