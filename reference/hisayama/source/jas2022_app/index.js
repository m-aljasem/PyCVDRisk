$(function() {
	$("#yesBtn").click(function() {
		document.getElementById("webOnlyQuestion").style.display = "none";
		document.getElementById("commonBeginButton").style.display = "inline";
	});

	$("#noBtn").click(function() {
		window.location.href = "../general/index.html";
	});

	$("#beginBtn").click(function() {
		window.location.href = "firstQuestion.html";
	});

	document.getElementById("commonBeginButton").style.display = "none";
});