/********************************************************************
*	Assign-offline.													*
*																	*
*		Richard Assar.												*
*																	*
/*******************************************************************/
/*
 Color animation jQuery-plugin
 http://www.bitstorm.org/jquery/color-animation/
 Copyright 2011 Edwin Martin <edwin@bitstorm.org>
 Released under the MIT and GPL licenses.
*/
(function(d){function i(){var b=d("script:first"),a=b.css("color"),c=false;if(/^rgba/.test(a))c=true;else try{c=a!=b.css("color","rgba(0, 0, 0, 0.5)").css("color");b.css("color",a)}catch(e){}return c}function g(b,a,c){var e="rgb"+(d.support.rgba?"a":"")+"("+parseInt(b[0]+c*(a[0]-b[0]),10)+","+parseInt(b[1]+c*(a[1]-b[1]),10)+","+parseInt(b[2]+c*(a[2]-b[2]),10);if(d.support.rgba)e+=","+(b&&a?parseFloat(b[3]+c*(a[3]-b[3])):1);e+=")";return e}function f(b){var a,c;if(a=/#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})/.exec(b))c=
[parseInt(a[1],16),parseInt(a[2],16),parseInt(a[3],16),1];else if(a=/#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])/.exec(b))c=[parseInt(a[1],16)*17,parseInt(a[2],16)*17,parseInt(a[3],16)*17,1];else if(a=/rgb\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*\)/.exec(b))c=[parseInt(a[1]),parseInt(a[2]),parseInt(a[3]),1];else if(a=/rgba\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9\.]*)\s*\)/.exec(b))c=[parseInt(a[1],10),parseInt(a[2],10),parseInt(a[3],10),parseFloat(a[4])];return c}
d.extend(true,d,{support:{rgba:i()}});var h=["color","backgroundColor","borderBottomColor","borderLeftColor","borderRightColor","borderTopColor","outlineColor"];d.each(h,function(b,a){d.fx.step[a]=function(c){if(!c.init){c.a=f(d(c.elem).css(a));c.end=f(c.end);c.init=true}c.elem.style[a]=g(c.a,c.end,c.pos)}});d.fx.step.borderColor=function(b){if(!b.init)b.end=f(b.end);var a=h.slice(2,6);d.each(a,function(c,e){b.init||(b[e]={a:f(d(b.elem).css(e))});b.elem.style[e]=g(b[e].a,b.end,b.pos)});b.init=true}})(jQuery);


// Flag to indicate when any value is out of usual range.
var outOfUsualRange = false;

// Simd score value, set on callback when each postcode area (EH9.js, etc.) array is loaded.
var simd;
var suffix;
var simdScore = 15.89;
var defaultSimd = true;
var unmatchedDigit = true;

// Stores which means are being used.
var meansFlags = {
	"tc" : {"flag" : false},
	"hdl" : {"flag" : false},
	"sbp" : {"flag" : false},
	"cpd" : {"flag" : false}
};

var valid_postcodes = ["QQ1","AB10","AB11","AB12","AB13","AB14","AB15","AB16","AB21","AB22","AB23","AB24","AB25","AB31","AB32","AB99","AB30","AB33","AB34","AB35","AB36","AB39","AB41","AB42","AB43","AB44","AB45","AB51","AB52","AB53","AB54","AB55","AB56","DD10","DD9","DD11","DD2","DD3","DD4","DD5","DD7","DD8","DG1","DG10","DG11","DG12","DG13","DG14","DG16","DG2","DG3","DG4","DG5","DG6","DG7","DG8","DG9","DD1","EH21","EH22","EH31","EH32","EH33","EH34","EH35","EH36","EH37","EH39","EH40","EH41","EH42","EH15","EH16","EH17","EH18","EH2","EH27","EH28","EH29","EH3","EH30","EH4","EH5","EH52","EH1","EH10","EH11","EH12","EH13","EH14","EH49","EH51","DD6","AB37","EH19","EH20","EH23","EH24","EH25","EH26","EH46","AB38","EH47","EH48","EH38","EH43","EH44","EH45","EH53","EH54","EH55","EH6","EH7","EH8","EH9","EH91","EH95","EH99","FK1","FK10","FK11","FK12","FK13","FK14","FK15","FK16","FK17","FK18","FK19","FK2","FK20","FK21","FK3","FK4","FK5","FK6","FK7","FK8","FK9","G1","G11","G12","G13","G14","G15","G2","G20","G21","G22","G23","G3","G31","G32","G33","G34","G4","G40","G41","G42","G43","G44","G45","G46","G5","G51","G52","G53","G58","G60","G61","G62","G63","G64","G65","G66","G67","G68","G69","G70","G71","G72","G73","G74","G75","G76","G77","G78","G79","G81","G82","G83","G84","G90","HS1","HS2","HS3","HS4","HS5","HS6","HS7","HS8","HS9","IV1","IV10","IV11","IV12","IV13","IV14","IV15","IV16","IV17","IV18","IV19","IV2","IV20","IV21","IV22","IV23","IV24","IV25","IV26","IV27","IV28","IV3","IV30","IV31","IV32","IV36","IV4","IV40","IV41","IV42","IV43","IV44","IV45","IV46","IV47","IV48","IV49","IV5","IV51","IV52","IV53","IV54","IV55","IV56","IV6","IV63","IV7","IV8","IV9","IV99","KA1","KA10","KA11","KA12","KA13","KA14","KA15","KA16","KA17","KA18","KA19","KA2","KA20","KA21","KA22","KA23","KA24","KA25","KA26","KA27","KA28","KA29","KA3","KA30","KA4","KA5","KA6","KA7","KA8","KA9","KW1","KW10","KW11","KW12","KW13","KW14","KW15","KW16","KW17","KW2","KW3","KW5","KW6","KW7","KW8","KW9","KY1","KY10","KY11","KY12","KY13","KY14","KY15","KY16","KY2","KY3","KY4","KY5","KY6","KY7","KY8","KY9","KY99","ML1","ML10","ML11","ML12","ML2","ML3","ML4","ML5","ML6","ML7","ML8","ML9","PA1","PA10","PA11","PA12","PA13","PA14","PA15","PA16","PA17","PA18","PA19","PA2","PA20","PA21","PA22","PA23","PA24","PA25","PA26","PA27","PA28","PA29","PA3","PA30","PA31","PA32","PA33","PA34","PA35","PA36","PA37","PA38","PA4","PA41","PA42","PA43","PA44","PA45","PA46","PA47","PA48","PA49","PA5","PA6","PA60","PA61","PA62","PA63","PA64","PA65","PA66","PA67","PA68","PA69","PA7","PA70","PA71","PA72","PA73","PA74","PA75","PA76","PA77","PA78","PA8","PA80","PA9","PH1","PH10","PH11","PH12","PH13","PH14","PH15","PH16","PH17","PH18","PH19","PH2","PH20","PH21","PH22","PH23","PH24","PH25","PH26","PH3","PH30","PH31","PH32","PH33","PH34","PH35","PH36","PH37","PH38","PH39","PH4","PH40","PH41","PH42","PH43","PH44","PH49","PH5","PH50","PH6","PH7","PH8","PH9","TD1","TD10","TD11","TD12","TD13","TD14","TD15","TD2","TD3","TD4","TD5","TD6","TD7","TD8","TD9","ZE1","ZE2","ZE3"];

/***
	toggleResultNotes()
		Toggles the notes displayed on the ASSIGN results.

***/

function toggleResultNotes() {
	if($("#score_notes_block").css("display") == "none") {
		$("#score_notes_block").css("display", "block");
	} else {
		$("#score_notes_block").css("display", "none");
	}
}

/***
	loaded()
		Called once the SIMD data is loaded, sets the global var.

***/

function loaded() {
	// In order to prevent any errors catch empty suffix...
	if(suffix != '') {
		// Check if the value is undefined.
		if(simd[suffix] == undefined) {
			unmatchedDigit = true;

			if(suffix.length == 3) {
				// Not found. Chop one character off the end of the suffix
				for(var key in simd) {
					if(key.indexOf(suffix.substr(0, 2)) == 0) {
						simdScore = (simd[key][0]);
						showNotice("postcode", "Unable to match postcode on final character. A tentative SIMD of " + simdScore + " has been calculated based on the remainder.");

						return;
					}
				}

				// Chop two characters off the end of the suffix
				for(var key in simd) {
					if(key.indexOf(suffix.substr(0, 1)) == 0) {
						simdScore = (simd[key][0]);
						showNotice("postcode", "Unable to match postcode on final two characters. A tentative SIMD of " + simdScore + " has been calculated based on the remainder.");

						return;
					}
				}

				// Fall back on default median value.
				simdScore = 20;
				showNotice("postcode", "Unable to match postcode sufficiently to allocate a specific SIMD code. Default median value 15.89 used for ASSIGN.");
				defaultSimd = true;

				return;
			} else if(suffix.length == 2) {
				// Chop.
				for(var key in simd) {
					if(key.indexOf(suffix.substr(0, 2)) == 0) {
						simdScore = (simd[key][0]);
						showNotice("postcode", "Unable to match postcode on final two characters. A tentative SIMD of " + simdScore + " has been calculated based on the remainder.");

						return;
					}
				}

				for(var key in simd) {
					if(key.indexOf(suffix.substr(0, 1)) == 0) {
						simdScore = (simd[key][0]);
						showNotice("postcode", "Unable to match postcode on final two characters. A tentative SIMD of " + simdScore + " has been calculated based on the remainder.");

						loadedCalled = true;
						return;
					}
				}

				for(var key in simd) {
					simdScore = (simd[key][0]);
					showNotice("postcode", "Unable to match postcode on final characters. A tentative SIMD of " + simdScore + " has been calculated based on the remainder.");

					return;
				}
			} else if(suffix.length == 1) {
				for(var key in simd) {
					simdScore = (simd[key][0]);
					showNotice("postcode", "Unable to match postcode on final  characters. A tentative SIMD of " + simdScore + " has been calculated based on the remainder.");

					return;
				}
			}
		} else { // Found it, so use it.
			simdScore = (simd[suffix][0]);
			showNotice("postcode", "Based on the postcode entered a SIMD score of " + simdScore + " has been calculated.");
			unmatchedDigit = false;
		}
	} else {
		// Fall back on default median value.
		simdScore = 20;
		showNotice("postcode", "Unable to match postcode sufficiently to allocate a specific SIMD code. Default median value 15.89 used for ASSIGN.");
		defaultSimd = true;
	}
}

/***
	calculate()
		The actual assign score calculation.

***/

function accMul(arg1, arg2)
{
	var m = 0;
	var s1 = arg1.toString ();
	var s2 = arg2.toString ();

	try { m += s1.split (".")[1].length } catch (e) {}
	try { m += s2.split (".")[1].length } catch (e) {}

	return Number(s1.replace(".","")) * Number(s2.replace(".","")) / Math.pow(10, m);
}

function accAdd(arg1, arg2) {
	var r1;
	var r2;
	var m;

	try { r1 = arg1.toString().split(".")[1].length } catch (e) {r1 = 0;}
	try { r2 = arg2.toString().split(".")[1].length } catch (e) {r2 = 0;}

	m = Math.pow(10, Math.max(r1, r2));

	return (arg1 * m + arg2 * m) / m;
}

function accDiv (arg1, arg2) {
	var t1 = 0;
	var t2 = 0;
	var r1;
	var r2;

	try {t1 = arg1.toString().split(".")[1].length} catch (e) {}
	try {t2 = arg2.toString().split(".")[1].length} catch (e) {}

	with (Math) {
		r1 = Number(arg1.toString().replace (".",""));
		r2 = Number(arg2.toString().replace (".",""));
		return (r1/r2) * pow(10, t2-t1);
	}
}
var calculating = false;
function calculate() {
	if(!calculating) {
		calculating = true;
		outOfUsualRange = false;
		var validated = true;

		$("input.numeric").each(function(item) {
			var thisValid = validate($(this).attr("name"), $(this).val());
			validated = validated && thisValid;
		});

		loadPostcodeRegion($("input[name='postcode']").val().toUpperCase()); // Required in case user skipped postcode entry.

		if(!validated)
			return false;

		// Get the inputs.
		var age = $("input[name='age']").val();
		var sex = $("input[name='sex']:checked").val();
		var postcode = $("input[name='postcode']").val();
		var chd = $("input[name='chd']:checked").val();
		var diabetes = $("input[name='diabetes']:checked").val();
		var arthritis = $("input[name='arthritis']:checked").val();
		var cpd = $("input[name='cpd']").val();
		var sbp = $("input[name='sbp']").val();
		var tc = $("input[name='tc']").val();
		var hdl = $("input[name='hdl']").val();

		if (arthritis > 0) {
			cpd = parseInt(cpd, 10) + 10;
		}

		// Linear combination, betas against inputs.
		var l = accAdd(
			accMul(betas[sex]["age"], age),
			accAdd(
				accMul(betas[sex]["tc"], tc),
				accAdd(
					accMul(betas[sex]["hdl"], hdl),
					accAdd(
						accMul(betas[sex]["sbp"], sbp),
						accAdd(
							accMul(betas[sex]["diabetes"], diabetes),
							accAdd(
								accMul(betas[sex]["chd"], chd),
								accAdd(
									accMul(betas[sex]["cpd"], cpd),
									accMul(betas[sex]["simd"], accDiv(simdScore, 10))
								)
							)
						)
					)
				)
			)
		);

		// Another linear combination, betas against coefficients.
		var l_bar =	accAdd(
			(accMul(betas[sex]["age"], beta_coefficients[sex]["age"])),
			accAdd(
				(accMul(betas[sex]["tc"], beta_coefficients[sex]["tc"])),
				accAdd(
					(accMul(betas[sex]["hdl"], beta_coefficients[sex]["hdl"])),
					accAdd(
						(accMul(betas[sex]["sbp"], beta_coefficients[sex]["sbp"])),
						accAdd(
							(accMul(betas[sex]["diabetes"], beta_coefficients[sex]["diabetes"])),
							accAdd(
								(accMul(betas[sex]["chd"], beta_coefficients[sex]["chd"])),
								accAdd(
									(accMul(betas[sex]["cpd"], beta_coefficients[sex]["cpd"])),
									(accMul(betas[sex]["simd"], beta_coefficients[sex]["simd"]))
								)
							)
						)
					)
				)
			)
		);

		// Gender specific base
		var base = { "male" : 0.8831, "female" : 0.9365 };

		// The ASSIGN score calculation.
		var a = l - l_bar;
		var b = Math.exp(a);
		var c = Math.pow(base[sex], b);
		var score = Math.round(accMul(100.0, accAdd(1.0, -c)));

		var meansUsed  = false;

		for(var j in meansFlags) {
			if(meansFlags[j]["flag"]) {
				meansUsed = true;
				break;
			}
		}


		// Display the results...
		var results = '<div><h3 style="width:100%;">ASSIGN Score: <span class="final_score">';

		results = results + score;

		if(outOfUsualRange || meansUsed || defaultSimd || unmatchedDigit) {
			results += '<sup id="asterix">*</sup>';
		}

		results = results + '</span></h3>';

		if(score < 20) {
			results += '<p class="score">ASSIGN score below 20 is not currently high risk, implying general preventive measures (nobody is free of risk) Repeat the scoring in 5 years or earlier, depending how high the current score is.</p>';
		} else {
			results += '<p class="score">ASSIGN 20 or more is high risk, implying specific medical preventive actions. No need to repeat the scoring on another occasion.</p>';
		}

		results += '<p id="score_info"><a id="score_info_toggle" onclick="toggleResultNotes(); return false;" href="#">notes</a></p>';

		results += '<p style="display: none;" id="score_notes_block">The ASSIGN number is the risk percent of cardiovascular disease over ten years, based on the recent SHHEC study. ASSIGN 20 means 20% risk. The actual risk is probably now less than that number. What matters more is what somebody\'s score is in relation to other ASSIGN scores, and therefore their priority or ranking for action to lower their risk.</p>';

		if(meansUsed || defaultSimd || unmatchedDigit) { results += "<p><sup>*</sup> In the absence of specific data for one or more variables mean population values have been used to calculate a provisional ASSIGN score. For a more specific score enter personal values for all variables, if you can, and recalculate.</p>"; }

		if(outOfUsualRange) {
			results = results + '<p><sup>*</sup> One or more values entered are outside the usual range for which the ASSIGN score was designed. Extreme values involve extrapolation so the score is tentative. Check your data entry first. Interpret such results with care.</p>';
		}



		//results += '<div class="reset-btn"><a href="../assign-score.html">New patient</a></div><input type="submit" class="submit_again" value="Recalculate" id="calc_submit"/></div>';



		$("#results-data").html(results);

		if($('#calc_submit').val() == 'Recalculate') {
			if($('div.gateway div#results h3').length > 0) {
				$('div.gateway div#results h3').css({'backgroundColor':'#00FF00'});
				$('div.gateway div#results h3').animate({'backgroundColor':'#FF0000'},700);
			}
		}

		$('#calc_submit').val('Recalculate');

		//$("#calc_submit").bind('mousedown', function() { calculate(); });
		calculating = false;
		return false;
	}
}



/***
	loadPostcodeRegion
		Callback for blur event of postcode field.

***/

function loadPostcodeRegion(postcode) {
	defaultSimd = false;

	var pc = postcode.replace(/\s/g, "");
	if (pc.length > 6) {
		postcode = pc.substr(0, 4) + ' ' + pc.substr(4);
	} else if (pc.length == 5) {
		postcode = pc.substr(0, 2) + ' ' + pc.substr(2);
	} else {
		postcode = pc.substr(0, 3) + ' ' + pc.substr(3);
	}


	var splitPostcode = postcode.split(" ");

	if(splitPostcode.length == 0) {
		if(postcode.length < 4) {
			simdScore = 20;
			showNotice("postcode", "Unable to match postcode sufficiently to allocate a specific SIMD code. Default median value 15.89 used for ASSIGN.");

			defaultSimd = true;
			return;
		} else if(postcode.length > 3 && postcode.length < 7) {
			region = postcode.substr(0, 3);
			suffix = postcode.substr(3, 3);
		} else if(postcode.length == 7) {
			region = postcode.substr(0, 4);
			suffix = postcode.substr(4, 3);
		} else {
			simdScore = 20;
			showNotice("postcode", "Unable to match postcode sufficiently to allocate a specific SIMD code. Default median value 15.89 used for ASSIGN.");

			defaultSimd = true;
			return;
		}
	} else {
		region = splitPostcode[0];
		suffix = splitPostcode[1];
	}

	$(".postcodeRegion").remove();

	try {
		delete simd;

		url = '../includes/js/postcode-simd/' + region + '.js';

		for (var i = 0; i < valid_postcodes.length; i += 1) {
			if (valid_postcodes[i] === region) {
				var match = true;
			}
		}

		if (!match) {
			simdScore = 20;
			showNotice("postcode", "Unable to match postcode sufficiently to allocate a specific SIMD code. Default median value 15.89 used for ASSIGN.");

			defaultSimd = true;
			return;
		}

		s = document.createElement('script');
      	s.src = url;
      	s.className = 'postcodeRegion';

      	document.body.appendChild(s);
	} catch(err) {
		simdScore = 20;
		showNotice("postcode", "Unable to match postcode sufficiently to allocate a specific SIMD code. Default median value 15.89 used for ASSIGN.");

		defaultSimd = true;
		return;
	}
}

/***
	 showNotice
	 	Shows notice on chosen field.

***/

function showNotice(risk_factor,message) {
	if(message) {
		var risk_factor_element = $("#" + risk_factor);

		var paragraph = risk_factor_element.children("div.expand").children("p.error");

		if($("#expand_" + risk_factor).css("display") == "block") {
			risk_factor_element.children("div.expand").attr("class", "expand_both");
		} else {
			risk_factor_element.children("div.expand").attr("class", "expand");
		}

		paragraph.css("display", "block");
		paragraph.text(message);

		paragraph.append('<span class="b"></span>');
		paragraph.append('<span class="t"></span>');

	} else {
		clearNotice(risk_factor);
	}
}

/***
	 clearNotice
	 	Clears notice on chosen field.

***/

function clearNotice(risk_factor) {
	var paragraph = $("#" + risk_factor + " p.error");

	$("#" + risk_factor).children("div.expand_both").attr("class", "expand");

	paragraph.css("display", "none");
	paragraph.attr("class", "error");
}

/***
	validate
		Validates a given input factor (age, cpd, hdl, etc...) and value
***/

function validate(factor, value) {
	if(value == '' ) { value = -1; }

	if(factor) {
		var lowerbound = ranges[factor]["lowerbound"];
		var upperbound = ranges[factor]["upperbound"];

		var usual_lowerbound = usual_ranges[factor]["lowerbound"];
		var usual_upperbound = usual_ranges[factor]["upperbound"];

		if((value < lowerbound) || (value > upperbound)) {
			showNotice(factor, "In order to return a score the " + ranges[factor]["name"] + " entered should be between " + lowerbound + " and " + upperbound + ".");
			return false;

		} else if((value < usual_lowerbound) || (value > usual_upperbound)) {
			outOfUsualRange = outOfUsualRange || true;

			showNotice(factor, "Outside the usual range of " + usual_lowerbound + "-" + usual_upperbound + " for " + ranges[factor]["name"] + ".");
			return true;

		} else {
			outOfUsualRange = outOfUsualRange || false;

			clearNotice(factor);
			return true;
		}
	}
}

/***
	Means

***/

// Fill in mean value and set flag to true.
function useMean(factor) {
	$("div#" + factor + " p.mean").text("use mean");

	meansFlags[factor]["flag"] = true;

	clearNotice(factor);
	setMean(factor);
}

// Re-activate anchor and set flag to false.
function resetMean(factor) {
	if(getMean(factor) != $("div#" + factor + " input").val()) {
		$("div#" + factor + " p.mean").html('<a href="#" onmousedown="useMean(\'' + factor + '\'); return false;">use mean</a>');

		meansFlags[factor]["flag"] = false;
	}

	validate(factor, $("input[name='" + factor + "']").val());
}

// Get the mean value for a given factor, depending on age and gender.
function getMean(factor) {
	var age = $("input[name='age']").val();

	for(var i = 0; i < means[factor].length; i++) {
		var lowerbound = means[factor][i]["lowerbound"];
		var upperbound = means[factor][i]["upperbound"];

		if((age >= lowerbound) && (age <= upperbound)) {
			clearNotice(factor);
			return means[factor][i][$("input[name='sex']:checked").val()];
		} else {
			if(meansFlags[factor]["flag"]) {
				showNotice(factor, "A mean value can not be calculated for the age and sex entered.");
			}
		}
	}

	return '';
}

// Set a mean value on a given factor
function setMean(factor) {
	var age = $("input[name='age']").val();

	$("input[name='" + factor + "']").val(getMean(factor));
}

// Update all means on all active factors.
function updateActiveMeans() {
	for(var j in meansFlags) {
		if(meansFlags[j]["flag"]) {
			setMean(j);
		}
	}
}

/***
	void main() { h0h0h0(); }

***/

$(document).ready(function() {
	// Load in simd on a refresh (when value persists from last session)
	if($("input[name='postcode']").val() != '') {
		loadPostcodeRegion($("input[name='postcode']").val().toUpperCase());
	}

	$("input[name='postcode']").blur(function() {
		$("#results-data").empty();
	});


	// Hide all the notes.
	$(".notes").hide();

	// Blur event for postcode field.
	$("div#postcode > div > input").blur(function() {
		//loadPostcodeRegion($(this).val().toUpperCase());
		loadPostcodeRegion($("input[name='postcode']").val().toUpperCase());
	});

	// Keypress event handler for numeric fields, allow refresh, backspace, numerals, and decimal point.
	$("input.numeric").keypress(function(key) {
		if(	(key.which >= 48 && key.which <= 57) ||
			key.which == 46 ||
			key.which == 8 ||
			key.which == 0
		) {
			return true;
		} else {
			return false;
		}
	}).blur(function() {	// Blur event for numerical fields.
		var value = $(this).val();

		// Get factor name
		var factor = $(this).parent().parent().attr("id");

		// validate
		validate(factor, value);

		if($(this).hasClass("hasMean")) {
			resetMean(factor);
		}
	});

	// Keypress event handler for postcode field, allow alphanumeric only.
	$("input[name='postcode']").keypress(function(key) {
		if(	(key.which >= 97 && key.which <= 122) ||
			(key.which >= 65 && key.which <= 90) ||
			(key.which >= 48 && key.which <= 57) ||
			key.which == 32 ||
			key.which == 8 ||
			key.which == 0
		) {
			return true;
		} else {
			return false;
		}
	});

	// Blur event for age input factor.
	$("input[name='age']").blur(function() {
		updateActiveMeans();
	});

	$("input[name='sex']").click(function() {
		updateActiveMeans();
	});

	// Notes togglers.
	$(".toggle_notes > a").click(function() {
		var parent = $(this).parent().parent().parent();

		var risk_factor = parent.attr("id");

		var element = $("#expand_" + risk_factor);

		if(element.css("display") == "block") {
			element.css("display", "none");

			if($("#" + risk_factor	+ " p.error").css("display") == "block") {
				$("#" + risk_factor).children("div.expand_both").attr("class", "expand");
			}
		} else {
			element.css("display", "block");

			if($("#" + risk_factor	+ " p.error").css("display") == "block") {
				$("#" + risk_factor).children("div.expand").attr("class", "expand_both");
			}
		}

		return false;
	});

	$("#calc_submit").click(function(e){
		e.preventDefault();
		calculate();
	});

});