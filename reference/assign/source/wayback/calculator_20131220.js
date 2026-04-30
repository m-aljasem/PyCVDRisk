/* Define globals */

	var active_algorithm;
	var score_algorithm;
	var simd;
	var xmlhttp;
	var lockedMeans = new Array();
	var non_optimum_values = new Array();
	var makingRequest = false;

/* Init the calculator */

	function initCalc() {

		/* Fetch active algorithm */
		score_algorithm = document.getElementById('algorithm').value;
		switchAlgorithm(score_algorithm);

		/* Toggle notes off */
		var allPageTags = new Array();
		var allPageTags=document.getElementsByTagName("P");
		for (i=0; i<allPageTags.length; i++) {
			if (allPageTags[i].className=="notes" || allPageTags[i].className=="notes wide") {
				allPageTags[i].style.display='none';
			}
		}

		/* Attach error handlers */
		var inputFields = document.getElementById('calculator').getElementsByTagName('input');
		for(i=0;i<inputFields.length;i++){
			if(inputFields[i].type=="text"){
				inputFields[i].onblur=function(){ validateInput(this,"updateSIMD"); updateMeans(); };
				inputFields[i].onchange=function(){clearLock(this);}
			} else if(inputFields[i].type == "radio") {
				inputFields[i].onchange=function(){updateMeans();}
			}
		}

		/* AJAX enable the form */
		document.getElementById('calculator').onsubmit=function() { getScores(this,'updateUI');  return false; }

		/* Update UI */
		var questions = document.getElementById('calculator').getElementsByTagName('div');
		for(i=0;i<questions.length;i++){
			if(questions[i].className=="question"){

				/* Update 'mean' links */
				var meanLinks = questions[i].getElementsByTagName("P");
				for(x=0;x<meanLinks.length;x++) {
					if(meanLinks[x].className=="mean") {
						meanLinks[x].firstChild.innerHTML = "use mean";
					}
				}

				/* Generate 'notes' links */
				var notes = document.createElement("P");
				notes.className="toggle_notes";
				var notes_link = document.createElement("A");
				notes_link.onclick=function(){ toggleNotes(this); return false; }
				notes_link.href="";
				notes_link.innerHTML="notes";
				notes.appendChild(notes_link);
				questions[i].appendChild(notes);
				document.getElementById('calc_submit').disabled=false;

			}
		}

	}

/* Toggle notes */

	function toggleNotes(nodeh) {

		element = document.getElementById("expand_"+nodeh.parentNode.parentNode.parentNode.id);
		parenth = document.getElementById(nodeh.parentNode.parentNode.parentNode.id);

		/* update the expand block's style if there is an active error */
		var paragraphs = parenth.getElementsByTagName('P');
		for(i=0;i<paragraphs.length;i++){
			if(paragraphs[i].className=="error" && paragraphs[i].style.display=="block"){
				paragraphs[i].parentNode.className="expand_both";
			}
		}

		/* expand the note */
		if(element.style.display=="block") {
			element.style.display="none";
			element.parentNode.className="expand";
		} else {
			element.style.display="block";
		}

	}

/* Switch score algorithm */

	function switchAlgorithm(algorithm){

		/* Set active algorithm in the form for PHP */
		document.getElementById('algorithm').value=algorithm;
		score_algorithm = algorithm;

		if(algorithm == "framingham") {
			document.getElementById("assign").style.display="block";
			document.getElementById("framingham").style.display="none";
			document.getElementById("lvh").style.display="block";
			document.getElementById("heart_score").setAttribute("class","heart_score_framingham");
			var framdisc = document.createElement('P');
			framdisc.id = "framinghamdisclaimer";
			document.getElementById('algorithm_list').parentNode.insertBefore(framdisc,document.getElementById('algorithm_list'));
			framdisc.innerHTML = "The Framingham cardiovascular score is provided here for comparison with the ASSIGN risk estimate and not to determine clinical management of patients. Those wishing to use the Framingham score should find an appropriate dedicated website.<br/>Note this is not the same as the JBS2 'Framingham' score.";
		} else {
			document.getElementById("assign").style.display="none";
			document.getElementById("framingham").style.display="block";
			document.getElementById("lvh").style.display="none";
			document.getElementById("heart_score").setAttribute('class','');
			if(document.getElementById("framinghamdisclaimer")) {
				document.getElementById("framinghamdisclaimer").parentNode.removeChild(document.getElementById("framinghamdisclaimer"));
			}
		}

		document.getElementById('calculator').action="estimate-the-risk/non-clinical/?algorithm="+algorithm;

		/* Update styles */
		var tabs=document.getElementById('algorithm_list').getElementsByTagName("LI");
		for(i=0;i<tabs.length;i++){ tabs[i].className=""; }
		document.getElementById(algorithm).className="active";

		if(active_algorithm == "assign" || active_algorithm == "framingham") {
			getScores(document.getElementById('calculator'),'updateUI');
		}

	}

/* Validate form input */
	var nextCall = new Array();
	var nextSIMD;
	function validateInput(input,evnt) {
		var message = "";
		var error = false;
		var resultsData = document.getElementById('results-data');

		if (resultsData) {
			resultsData.parentElement.removeChild(resultsData);
		}

		document.getElementById('heart_score').style.background ='';
		document.getElementById('heart_score').innerHTML = '';

		/* Fetch simd */
		if(input.name=="postcode" || input.name=="simd") {
			if(input.name=="simd") {
				if(input.value.length>0 && (input.value < 0.3 || input.value > 90)) {
					message="In order to return a score the SIMD entered should be between 0.3 and 90.";
					error = true;
				} else if(input.value.length>0 && isNaN(parseInt(input.value))) {
					message="In order to return a score the SIMD entered should contain only numerical data.";
					error = true;
				} else if(input.value.length>0) {
					if(evnt=="updateSIMD") { getScores(document.getElementById('calculator'),"updateSIMD"); }
				}
			} else {
				if(document.getElementById("simd_manual")) {
					var temp_simd = document.getElementById("simd_manual").value;
				} else { temp_simd = ""; }
				if(evnt=="updateSIMD" && (temp_simd.length<=0 || ((temp_simd > 0.3 && temp_simd < 90) && !isNaN(parseInt(temp_simd))))) {
					getScores(document.getElementById('calculator'),"updateSIMD");
				}
			}
		}

		/* check ranges */
		if(ranges[input.name]) {
			if(input.value < ranges[input.name][0] || input.value > ranges[input.name][1]) {
				if(ranges[input.name][2]=="Current age") ranges[input.name][2] = "age";
				message="In order to return a score the "+ranges[input.name][2]+" entered should be between "+ranges[input.name][0]+" and "+ranges[input.name][1]+".";
				error = true;
			}
		}

		/* check notices */
		if(notices[input.name]) {
			if(!message && (input.value < notices[input.name][0] || input.value > notices[input.name][1])) {
				message="Outside the usual range of "+notices[input.name][0]+"-"+notices[input.name][1]+" for "+notices[input.name][2]+".";
				non_optimum_values[non_optimum_values.length] = input.name;
			} else {
				/* splice the value out, if it's there */
				for(f=0;f<non_optimum_values.length;f++) {
					if(non_optimum_values[f]==input.name) {
						non_optimum_values.splice(f,1);
					}
				}
			}
		}

		/* Check validity */
		if(isNaN(parseInt(input.value)) && input.name!="postcode" && input.name!="simd" && input.name!="name") {
			message="In order to return a score the "+ranges[input.name][2]+" entered should contain only numerical data.";
			error = true;
		}

		if(message.length>0) { nextCall[nextCall.length+1] = "showNotice('"+input.name+"','"+message+"')"; if(input.name=="simd") { nextCall[nextCall.length+1] = 'clearNotice("postcode")'; } } else { nextCall[nextCall.length+1]= "clearNotice('"+input.name+"')"; };

		setTimeout("parseValidation()",400);
		if(error) { return message; }

	}

	function parseValidation() {
		for(i=0;i<nextCall.length;i++) {
			eval(nextCall[i]);
		} nextCall= new Array(); eval(nextSIMD); nextSIMD = "";
	}

/* Show notice */

	function showNotice(risk_factor,message) {

		var paragraphs = document.getElementById(risk_factor).getElementsByTagName('P');
		for(k=0;k<paragraphs.length;k++){
			if(paragraphs[k].className=="error") {
				if(message) {

					/* update the expanded block's style if there is an active note */
					if(document.getElementById('expand_'+risk_factor).style.display=="block") {
						var divs = document.getElementById(risk_factor).getElementsByTagName('DIV');
						for(x=0;x<divs.length;x++) {
							if(divs[x].className=="expand") divs[x].className="expand_both";
						}
					}

					/* expand the error block and insert message */
					paragraphs[k].style.display="block";
					paragraphs[k].innerHTML=message;

					/* add some corners */
					b = document.createElement('span'); b.className="b";
					t = document.createElement('span'); t.className="t";
					paragraphs[k].appendChild(t); paragraphs[k].appendChild(b);

				} else {
					clearNotice(risk_factor);
				}
			}
		}
	}

/* Clear notice */

	function clearNotice(risk_factor) {
		if(document.getElementById(risk_factor)) {
			var paragraphs = document.getElementById(risk_factor).getElementsByTagName('P');
			for(x=0;x<paragraphs.length;x++){
				if(paragraphs[x].className=="error") {
					paragraphs[x].style.display="none";
					paragraphs[x].parentNode.className="expand";
					paragraphs[x].innerHTML="";
				}
			}
		}
	}


/* Insert mean values */

	function useMean(node) {

		if(typeof(node)=="object") {
			var question = node.parentNode.parentNode.parentNode.id;
		} else {
			var question = node;
		}

		/* Fetch age and sex */
		var age = document.getElementById('age').getElementsByTagName('input')[0].value;
		var sexes = document.getElementById('sex').getElementsByTagName('input');
		for(x=0;x<sexes.length;x++) {
			if(sexes[x].checked==true){
				var sex = sexes[x].value;
			}
		}

		/* Check the current age and fetch the respective range */
		var means_ranges=means[question];
		var means_array = new Array();
		for(i=0;i<means_ranges.length;i++){
			if(age>=means_ranges[i][0] && age<=means_ranges[i][1]) {
				means_array = means_ranges[i];
			}
		}

		/* set the mean value, depending on gender */
		if(sex=="m" && means_array.length>0) {
			document.getElementById(question).getElementsByTagName('input')[0].value=means_array[2];
			nextCall[nextCall.length+1] = 'clearNotice("'+question+'")';
		} else if(sex=="f" && means_array.length>0) {
			document.getElementById(question).getElementsByTagName('input')[0].value=means_array[3];
			nextCall[nextCall.length+1] = 'clearNotice("'+question+'")';
		} else {
			nextCall[nextCall.length+1] = 'showNotice("'+question+'","A mean value can not be calculated for the age and sex entered.")';
			document.getElementById(question).getElementsByTagName('input')[0].value="";
		}

		setTimeout("parseValidation()",300);

		var found = false;
		for(f=0;f<lockedMeans.length;f++) { if(lockedMeans[f]==question) { found = true; }}
		if(found==false) { lockedMeans[lockedMeans.length]=question; }

		var paragraphs = document.getElementById(question).getElementsByTagName('p');
		for(pm=0;pm<paragraphs.length;pm++) {
			if(paragraphs[pm].className=="mean") {
				paragraphs[pm].innerHTML="using mean";
			}
		}

	}

/* update means for locked fields */

	function updateMeans() {
		for(u=0;u<lockedMeans.length;u++) {
			useMean(lockedMeans[u]);
		}
	}

/* clear mean lock on field */

	function clearLock(node) {
		var question = node.parentNode.parentNode.id;

		var found = false;
		for(f=0;f<lockedMeans.length;f++) {
			if(lockedMeans[f]==question) {
				lockedMeans.splice(f,1);
				var paragraphs = document.getElementById(question).getElementsByTagName('p');
				for(pm=0;pm<paragraphs.length;pm++) {
					if(paragraphs[pm].className=="mean") {
						paragraphs[pm].innerHTML='<a href="estimate-the-risk/risk-factors/" onclick="useMean(this); return false;">use mean</a>';
					}
				}
			}
		}
	}

/* AJAX call to calculation class */

	function getScores(form,evnt) {

		/* build parameter list */
		var parameters="";
		var delim="";
		var errors = new Array();
		var inputs = form.getElementsByTagName("input");

		for(i=0;i<inputs.length;i++){
			if(inputs[i].type=="radio") {
				if(inputs[i].checked) {
					parameters = parameters+delim+inputs[i].name+"="+encodeURI(inputs[i].value);
					delim = "&";
				}
			} else {
				if(evnt=="updateUI" && inputs[i].type == "text") {
					var check = validateInput(inputs[i],evnt);
					if(check) { errors[i] = check; }
				}
				if(inputs[i]) {
					parameters = parameters+delim+inputs[i].name+"="+encodeURI(inputs[i].value);
					delim = "&";
				}
			}
		}

		if(errors.length<1) {
			/* setup xmlhttp object */

			if (!makingRequest) {
				xmlhttp=null
				makingRequest = true;

				if (window.XMLHttpRequest){
					xmlhttp=new XMLHttpRequest()
				} else if (window.ActiveXObject) {
					xmlhttp=new ActiveXObject("Microsoft.XMLHTTP")
				}
				/* send the request */
				if (xmlhttp!=null) {
					/* prevent caching in IE6 */
					nocache = new Date;
					if(evnt=="updateUI") xmlhttp.onreadystatechange=updateUI;
					if(evnt=="updateSIMD") xmlhttp.onreadystatechange=updateSIMD;
					xmlhttp.open('POST', "/includes/php/preview-calculator.ajax.php?evnt="+evnt+"&nocache="+nocache.getTime(), true);
		      		xmlhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
		      		xmlhttp.setRequestHeader("Content-length", parameters.length);
		      		xmlhttp.setRequestHeader("Connection", "close");
					xmlhttp.send(parameters);
				} else {
					alert("Sorry, your browser does not support this feature.\n Please disable Javascript and try again.")
				}
			}
		} else {
			var resultsBlock = document.getElementById('results');

			document.getElementById('heart_score').style.background="url(images/heart-dynamic.png) no-repeat";
			document.getElementById('heart_score').innerHTML="Score Calculator";
			var resultUrl = window.location.pathname;
			resultsBlock.innerHTML = "</div><input id=\"calc_submit\" type=\"submit\" value=\"Recalculate\" class=\"submit_again\" /><div class=\"print-btn\"><a href='#' onclick=\"window.print();return false;\">Print</a></div><div class=\"print-btn new-patient\"><a href='"+resultUrl+"'>New patient</a></div>";
		}

	}

/* Update the UI with results */

	function updateUI() {
		var html;

		if (xmlhttp.readyState==1) {
			if(document.getElementById('ajax_loading')) {
				document.getElementById('ajax_loading').parentNode.removeChild(document.getElementById('ajax_loading'));
			}

			var spinner = document.createElement('span');
			spinner.id = "ajax_loading";
			document.getElementById('results').appendChild(spinner);
			document.getElementById('calc_submit').style.color="#666666";
			document.getElementById('calc_submit').disabled="false";

		}

		if (xmlhttp.readyState==4) {
			/* Update SIMD */

			var manual_simd = xmlhttp.responseXML.getElementsByTagName("MANUAL")[0].firstChild.nodeValue;
			var manual_simd_score = xmlhttp.responseXML.getElementsByTagName("SIMD")[0].firstChild.nodeValue;
			var default_simd = xmlhttp.responseXML.getElementsByTagName("DEFAULT")[0].firstChild.nodeValue;

			if(manual_simd == "true") {
				var message = "A SIMD score of "+manual_simd_score+" has been used. Clear the SIMD field, above, if you wish to amend the Postcode and calculate a new SIMD.";
				if(message) { nextSIMD = "showNotice('simd','"+message+"'); clearNotice('postcode')"; } else { nextSIMD = "clearNotice('"+input.name+"')"; };
				/* Google analytics */
				if(document.getElementById('calc_submit').value=="Calculate") {
					if(pageTracker) { pageTracker._trackPageview("/calculator/assignvisitor/calculate/"); }
				} else {
					if(pageTracker) { pageTracker._trackPageview("/calculator/assignvisitor/recalculate/"); }
				}
			} else {
				var message = xmlhttp.responseXML.getElementsByTagName("NOTICE")[0].firstChild.nodeValue;
				if(document.getElementById("simd_manual")) {
					if(default_simd == "false") { document.getElementById("simd_manual").value=manual_simd_score; }
				}
				if(message) { nextSIMD = "showNotice('postcode','"+message+"'); clearNotice('simd')"; } else { nextSIMD = "clearNotice('"+input.name+"')"; };
				/* Google analytics */
				if(document.getElementById('calc_submit').value=="Calculate") {
					if(pageTracker) { pageTracker._trackPageview("/calculator/nhs/calculate/"); }
				} else {
					if(pageTracker) { pageTracker._trackPageview("/calculator/nhs/recalculate/"); }
				}
			}

			/* Update UI */
			var resultsBlock = document.getElementById('results');

			active_algorithm = xmlhttp.responseXML.getElementsByTagName("ALGORITHM")[0].firstChild.nodeValue;
			var score_assign = xmlhttp.responseXML.getElementsByTagName("ASSIGN")[0].firstChild.nodeValue;
			var score_framingham = xmlhttp.responseXML.getElementsByTagName("FRAMINGHAM")[0].firstChild.nodeValue;

			html = '<div id="results-data">';

			if(lockedMeans.length > 0 || non_optimum_values.length > 0 || default_simd == "true") { var denote = "*"; } else { var denote = ""; }

			if(active_algorithm == "assign") {

				/* Update heart */
				document.getElementById('heart_score').style.background="url('includes/php/dynamic_heart.php?score="+score_assign+"') 18px 7px no-repeat";
				document.getElementById('heart_score').innerHTML = '<span id="bannerheart"></span><span class="alg">ASSIGN score is <span id="final">'+score_assign+'</span>';
				document.getElementById('final').innerHTML=score_assign;

				html+="<h3>ASSIGN Score: <span class=\"final_score\">"+score_assign+"<sup>"+denote+"</sup></span></h3>";

				if(score_assign>=20) {
					html+="<p class=\"score\">ASSIGN 20 or more is high risk, implying specific medical preventive actions. No need to repeat the scoring on another occasion.</p>";
				} else {
					html+="<p class=\"score\">ASSIGN score below 20 is not currently high risk, implying general preventive measures (nobody is free of risk) Repeat the scoring in 5 years or earlier, depending how high the current score is.</p>";
				}

				var score_notes = document.createElement('p');
				score_notes.id = "score_info";

				var score_notes_link = document.createElement('a');
				score_notes_link.id = "score_info_toggle";
				score_notes_link.setAttribute("onclick","sigh(); return false;");
				score_notes_link.setAttribute("href","about/beginners/");
				score_notes_link.innerHTML="notes";

				score_notes.appendChild(score_notes_link);
				document.getElementById('results').appendChild(score_notes);

				html+="<p id=\"score_notes_block\" style=\"display: none\">The ASSIGN number is the risk percent of cardiovascular disease over ten years, based on the recent SHHEC study. ASSIGN 20 means 20% risk. The actual risk is probably now less than that number. What matters more is what somebody&#39;s score is in relation to other ASSIGN scores, and therefore their priority or ranking for action to lower their risk.</p>";
				if(lockedMeans.length > 0 || default_simd == "true") html+="<p><sup>*</sup> In the absence of specific data for one or more variables mean population values have been used to calculate a provisional ASSIGN score. For a more specific score enter personal values for all variables, if you can, and recalculate.</p>";
				if(non_optimum_values.length > 0) { html+="<p><sup>*</sup> One or more values entered are outside the usual range for which the ASSIGN score was designed. Extreme values involve extrapolation so the score is tentative. Check your data entry first. Interpret such results with care.</p>"; }

			} else if(active_algorithm == "framingham") {

				/* Update heart */
				document.getElementById('heart_score').style.background="url('includes/php/dynamic_heart.php?score="+score_framingham+"') 18px 7px no-repeat";
				document.getElementById('heart_score').innerHTML = '<span id="bannerheart"></span>Framingham score is <span id="final">'+score_framingham+'</span>';
				document.getElementById('final').innerHTML=score_framingham;
				document.getElementById('bannerheart').style.backgroundImage="url(images/calculate-banner-framingham.png)";

				html+="<h3>Framingham Score: <span class=\"final_score\">"+score_framingham+"<sup>"+denote+"</sup></span></h3>";
				html+="<p class=\"score\">Based on the personal details entered above, Framingham estimates a "+score_framingham+"% chance of cardiovascular disease developing in the next 10 years.</p><p>Note, this calculation is only for comparison with ASSIGN, and is not the same Framingham score as used by JBS2.</p>";
				if(lockedMeans.length > 0 || default_simd == "true") html+="<p><sup>*</sup> In the absence of specific data for one or more variables mean population values have been used to calculate a provisional Framingham score. For a more specific score enter personal values for all variables, if you can, and recalculate.</p>";
				if(non_optimum_values.length > 0) { html+="<p><sup>*</sup> One or more values entered are outside the usual range for which the ASSIGN score was designed. Extreme values involve extrapolation so the score is tentative. Check your data entry first. Interpret such results with care.</p>"; }
			}
			var resultUrl = window.location.pathname;

			html+="</div><input id=\"calc_submit\" type=\"submit\" value=\"Recalculate\" class=\"submit_again\" />";
			html+="<div class=\"print-btn\"><a href='#' onclick=\"window.print();return false;\">Print</a></div>";
			html+="<div class=\"print-btn new-patient\"><a href='"+resultUrl+"'>New patient</a></div>";

			resultsBlock.innerHTML = html;

			non_optimum_values = new Array();
			makingRequest = false;
			parseValidation();
		}

	}

	function sigh(){
		if(document.getElementById('score_notes_block').style.display=="block") {
			document.getElementById('score_notes_block').style.display="none";
		} else {
			document.getElementById('score_notes_block').style.display="block";
		}
	}

	function updateSIMD() {
		if (xmlhttp.readyState==1) {
			/*display loading*/
		}
		if (xmlhttp.readyState==4) {
			makingRequest = false;
			/* Update SIMD */
			simd = xmlhttp.responseXML.getElementsByTagName("SIMD")[0].firstChild.nodeValue;
			var manual_simd = xmlhttp.responseXML.getElementsByTagName("MANUAL")[0].firstChild.nodeValue;

			if(manual_simd == "true") {
				var message = "A SIMD score of "+simd+" has been used. Clear the SIMD field, above, if you wish to amend the Postcode and calculate a new SIMD.";
				if(message) { nextSIMD = "showNotice('simd','"+message+"'); clearNotice('postcode')"; } else { nextSIMD = "clearNotice('"+input.name+"')"; };
			} else {
				var message = xmlhttp.responseXML.getElementsByTagName("NOTICE")[0].firstChild.nodeValue;
				if(document.getElementById("simd_manual")) {
					document.getElementById("simd_manual").value=simd;
					nextCall[nextCall.length+1] = 'showNotice("simd","Clear the SIMD field, above, if you wish to amend the Postcode and calculate a new SIMD.")';
				}
				if(message) {
					if(nextCall.length>0) {
						nextSIMD = "showNotice('postcode','"+message+"');";
					} else {
						nextSIMD = "showNotice('postcode','"+message+"'); clearNotice('simd')";
					}
				} else {
					 nextSIMD = "clearNotice('"+input.name+"')";
				}
			}

			setTimeout("parseValidation()",400);

		}
	}
