/**
 *
 */
function showAlert(msg)
{	
	var u = window.navigator.userAgent.toLowerCase();	
	if(u.indexOf("ipad") != -1 || u.indexOf("iphone") != -1 || u.indexOf("ipod") != -1)
	{
		document.location.href = "app-alert://ご確認ください/" + msg;
	}
	else if(u.indexOf("android") != -1)
	{
		document.location.href = "app-alert://android/ご確認ください/" + msg;
	}
	else
	{
		alert("ご確認ください" + "\n" + msg);
	}
}

/**
 * iframe以外で表示されている場合(Web版のみ必要)
 */
$(function() {
	if(window == window.parent) {
		location.href="https://www.j-athero.org/jp/general/ge_tool2/";
	}
});

/**
 * iOSでキーボード表示時にfixedなelementがふっとぶ問題対策
 */
$(function()
{
	var u = window.navigator.userAgent.toLowerCase();
	if(u.indexOf("ipad") != -1 || u.indexOf("iphone") != -1 || u.indexOf("ipod") != -1)
	{
		$(document)
		.on('focus', 'input[type=number]', function() {
			$(".headerRoot").addClass('hideHeader');
			$(".footerRoot").addClass('hideHeader');
		})
		.on('blur', 'input[type=number]', function() {
			$(".headerRoot").removeClass('hideHeader');
			$(".footerRoot").removeClass('hideHeader');
		});
	}
});

/**
 * URLに含まれるパラメータを取得する
 * @return URLを解析した結果のハッシュ
 */
function getCGIParam()
{
	//console.log(location.search);
	
	var param = new Object();
	if(location.search.length >= 2 && location.search.substring(0, 1) == '?')
	{
		var b = location.search.substring(1).split('&');
		if(b != '')
		{
		    for(var i = 0; i < b.length; i++)
		    {
		    	var c = b[i].split('=');
		    	if(c != '' && c.length >= 2)
		    	{
			    	param[decodeURI(c[0])] = decodeURIComponent(c[1].replace(/\+/g, ' '));		    		
		    	}
		    }
		}
	}

	return param;
}
