/**
 *
 */
function createResult2(sum) {
	const rate_list = [
		[7,		1.0,	0,	"1.0%以下",	"低リスク"],
		[8,		1.1,	0,	"1.1%",		"低リスク"],
		[9,		1.3,	0,	"1.3%",		"低リスク"],
		[10,	1.4,	0,	"1.4%",		"低リスク"],
		[11,	1.7,	0,	"1.7%",		"低リスク"],
		[12,	1.9,	0,	"1.9%",		"低リスク"],
		[13,	2.2,	1,	"2.2%",		"中リスク"],
		[14,	2.6,	1,	"2.6%",		"中リスク"],
		[15,	3.0,	1,	"3.0%",		"中リスク"],
		[16,	3.4,	1,	"3.4%",		"中リスク"],
		[17,	3.9,	1,	"3.9%",		"中リスク"],
		[18,	4.5,	1,	"4.5%",		"中リスク"],
		[19,	5.2,	1,	"5.2%",		"中リスク"],
		[20,	6.0,	1,	"6.0%",		"中リスク"],
		[21,	6.9,	1,	"6.9%",		"中リスク"],
		[22,	7.9,	1,	"7.9%",		"中リスク"],
		[23,	9.1,	1,	"9.1%",		"中リスク"],
		[24,	10.4,	2,	"10.4%",	"高リスク"],
		[25,	11.9,	2,	"11.9%",	"高リスク"],
		[26,	13.6,	2,	"13.6%",	"高リスク"],
		[27,	15.5,	2,	"15.5%",	"高リスク"],
		[28,	17.7,	2,	"17.7%",	"高リスク"],
		[29,	20.2,	2,	"20.2%",	"高リスク"],
		[30,	22.9,	2,	"22.9%",	"高リスク"],
		[31,	25.9,	2,	"25.9%",	"高リスク"],
		[32,	29.3,	2,	"29.3%",	"高リスク"],
		[33,	33.0,	2,	"33.0%",	"高リスク"],
		[34,	37.0,	2,	"37.0%",	"高リスク"],
		[35,	41.4,	2,	"41.4%",	"高リスク"],
		[36,	46.1,	2,	"46.1%",	"高リスク"],
		[99999,	50.0,	2,	"50%以上",	"高リスク"]];

	let result = {};
	
	for(let i = 0; i < rate_list.length; i++) {
		let rate_item = rate_list[i];
		if(sum <= rate_item[0]) {
			result.score = sum;
			result.rate = rate_item[1];
			result.risk = rate_item[2];
			result.rateText = rate_item[3];
			result.riskText = rate_item[4];
			break;
		}
	}
	
	return result;
}

/**
 *
 */
function calsSuitaScore2() {
	let score_sum = 0;
	let score_sum_min = 0;

	// ----------------------------------------
	// 年齢

	let old_val = $("[name=old]:checked").val();

	if(typeof old_val == 'undefined' || old_val.length < 1 || old_val.match(/[^0-9]+/)) {
		showAlert("【年齢】を選択してください。");
		return false;
	}

	if(old_val > 79) {
		showAlert("【年齢】を選択してください。");
		return false;
	} else if(old_val >= 70) {
		score_sum += 16;
		score_sum_min += 16;
	} else if(old_val >= 60) {
		score_sum += 11;
		score_sum_min += 11;
	} else if(old_val >= 50) {
		score_sum += 5;
		score_sum_min += 5;
	} else if(old_val >= 40) {
		score_sum += 0;
		score_sum_min += 0;
	} else {
		showAlert("【年齢】を選択してください。");
		return false;
	}

	// ----------------------------------------
	// 性別

	let sex_val = $("[name=sex]:checked").val();

	if(typeof sex_val == 'undefined' || sex_val.length < 1) {
		showAlert("【性別】を選択してください。");
		return false;
	}

	score_sum += parseInt(sex_val, 10);
	score_sum_min += parseInt(sex_val, 10);

	// ----------------------------------------
	// HDLコレステロール

	let hdl_c_val = $("[name=hdl]").val();

	if(typeof hdl_c_val == 'undefined' ||
			hdl_c_val.length < 1 ||
			hdl_c_val.match(/[^0-9]+/)) {
		showAlert("【HDL-C】を半角数字で入力してください。");
		return false;
	}
	
	if(hdl_c_val >= 60) {
		score_sum += 0;
	} else if(hdl_c_val >= 40) {
		score_sum += 1;
	} else {
		score_sum += 2;
	}
	
	score_sum_min += 0;

	// ----------------------------------------
	// 総コレステロール・LDLコレステロール・トリグリセライド

	let tc_val = $("[name=tc]").val();
	let ldl_c_val = $("[name=ldl]").val();
	let tg_val = $("[name=tg]").val();
	let non_hdl_c_val;

	// LDL-Cが未入力の場合
	if(typeof ldl_c_val == 'undefined' || ldl_c_val.length < 1 || ldl_c_val.match(/[^0-9]+/)) {

		// TCの入力チェック
		if(typeof tc_val == 'undefined' || tc_val.length < 1 || tc_val.match(/[^0-9]+/)) {
			showAlert("【総コレステロール(TC)】又は【LDL-C】を半角数字で入力してください。");
			return false;
		}	

		// TGの入力チェック
		if(typeof tg_val == 'undefined' || tg_val.length < 1 || tg_val.match(/[^0-9]+/)) {
			showAlert("【TG】又は【LDL-C】を半角数字で入力してください。");
			return false;
		}	

		// TCとHDL-CとTGからLDL-Cを算出する
		ldl_c_val = parseInt(tc_val - hdl_c_val - (tg_val / 5), 10);
		if(ldl_c_val < 1) {
			showAlert("【総コレステロール(TC)】【HDL-C】【TG】の値が正しく入力されているか見直してください。");
			return false;
		}
	}

	// TCが入力されているとき
	if(!(typeof tc_val == 'undefined' || tc_val.length < 1 || tc_val.match(/[^0-9]+/))) {
		// Non-HDL-CをTCとHDL-Cから算出する
		non_hdl_c_val = parseInt(tc_val - hdl_c_val, 10);
		if(non_hdl_c_val < 1) {
			showAlert("【総コレステロール(TC)】【HDL-C】の値が正しく入力されているか見直してください。");
			return false;			
		}
	} else {
		tc_val = -1;
		non_hdl_c_val = -1;
	}

	// TGが入力されていないとき
	if(typeof tg_val == 'undefined' || tg_val.length < 1 || tg_val.match(/[^0-9]+/)) {
		tg_val = -1;
	}

	if(ldl_c_val >= 160) {
		score_sum += 3;
	} else if(ldl_c_val >= 140) {
		score_sum += 2;
	} else if(ldl_c_val >= 120) {
		score_sum += 1;
	} else {
		score_sum += 0;
	}

	score_sum_min += 0;

	// ----------------------------------------
	// 血圧

	let sbp_val = $("[name=systolicBloodPressure]").val();

	if(typeof sbp_val == 'undefined' ||
			sbp_val.length < 1 ||
			sbp_val.match(/[^0-9]+/)) {
		showAlert("【収縮期血圧】を半角数字で入力してください。");
		return false;
	}

	if(sbp_val >= 160) {
		score_sum += 4;
	} else if(sbp_val >= 140) {
		score_sum += 3;
	} else if(sbp_val >= 130) {
		score_sum += 2;
	} else if(sbp_val >= 120) {
		score_sum += 1;
	} else {
		score_sum += 0;
	}

	score_sum_min += 0;

	// ----------------------------------------
	// 耐糖能異常

	let dm_val = $("[name=diabeteMellitus]:checked").val();

	if(typeof dm_val == 'undefined' || dm_val.length < 1) {
		showAlert("【糖代謝異常】を選択してください。");
		return false;
	}

	score_sum += parseInt(dm_val, 10);
	score_sum_min += 0;

	// ----------------------------------------
	// 喫煙習慣

	var smoking_val = $("[name=smoking]:checked").val();

	if(typeof smoking_val == 'undefined' || smoking_val.length < 1) {
		showAlert("【喫煙の習慣】を選択してください。");
		return false;
	}

	score_sum += parseInt(smoking_val, 10);
	score_sum_min += 0;

	let res = createResult2(score_sum);
	let res_min = createResult2(score_sum_min);

	//console.log(res);
	//console.log(res_min);

	let param = "";
	param += "?rate=" + encodeURIComponent(res.rate);
	param += "&rateText=" + encodeURIComponent(res.rateText);
	param += "&risk=" + encodeURIComponent(res.risk);
	param += "&riskText=" + encodeURIComponent(res.riskText);
	param += "&rateMin=" + encodeURIComponent(res_min.rate);

	let old_dict = {"45":"４０～４９", "55":"５０～５９", "65":"６０～６９", "75":"７０～７９"};
	param += "&old=" + encodeURIComponent(old_dict[old_val]);

	let sex_dict = {"0":"女性", "7":"男性"};
	param += "&sex=" + encodeURIComponent(sex_dict[sex_val]);

	let smoking_dict = {"0":"ない", "2":"ある"};
	param += "&smoking=" + encodeURIComponent(smoking_dict[smoking_val]);

	param += "&ldl=" + encodeURIComponent(ldl_c_val);
	param += "&hdl=" + encodeURIComponent(hdl_c_val);
	param += "&systolicBloodPressure=" + encodeURIComponent(sbp_val);

	let dm_dict = {"0":"なし または 不明", "1":"あり"};
	param += "&diabeteMellitus=" + encodeURIComponent(dm_dict[dm_val]);

	param += "&tc=" + encodeURIComponent(tc_val);
	param += "&tg=" + encodeURIComponent(tg_val);
	param += "&non_hdl_c=" + encodeURIComponent(non_hdl_c_val);

	//alert(param);

	return param;
}

$(function() {
	// Enterによる誤submitを抑制
	$(document).on("keypress", "input:not(.allow_submit)", function(event) {
		if(event.which !== 13) {
			return true;
		} else {
			// スマホキーボードでEnterがsubmitを期待させる表記になるので単に入力を無視するのではなく
			// フォーカス外してキーボードを隠す
			this.blur();
			return false;
		}
	});
	// 診断ボタンの動作
	$("#goBtn").click(function() {
		let param = calsSuitaScore2();
		if(typeof param == 'string' && param.length > 0) {
			window.location.href = "result.html" + param;
		}
		return false;
	});
	// TCを変更した時の処理
	$("[name=tc]").change(function() {
		let ldl_val = $("[name=ldl]").val();
		let tc_val = $("[name=tc]").val();
		if(tc_val.length > 0 && ldl_val.length > 0) {
			showAlert("「総コレステロール(TC)」が入力された場合、「LDL-C」は使用しません。「LDL-C」を空欄にします。");
			$("[name=ldl]").val("");
		}
	});
	// LDL-Cを変更した時の処理
	$("[name=ldl]").change(function() {
		let ldl_val = $("[name=ldl]").val();
		let tc_val = $("[name=tc]").val();
		if(tc_val.length > 0 && ldl_val.length > 0) {
			showAlert("「総コレステロール(TC)」が入力されている場合、「LDL-C」は使用しません。");
			$("[name=ldl]").val("");
		}
	});
});