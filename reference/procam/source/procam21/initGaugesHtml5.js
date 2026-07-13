/***************************************************************
*  Copyright notice
*  (c) 2012 PrevaMed GmbH <info@prevamed.com>
*  All rights reserved.
***************************************************************/

//gauges

var _gaugeStroke = renderGauge('gauge_stroke', 0, 100, 50, 0, 0, 50, 100);
var _gaugeQuick = renderGauge('gauge_quick', 0, 100, 33, 33, 66, 66, 100);
var _gaugeHealth = renderGauge('gauge_health', 0, 100, 33, 33, 66, 66, 100);


//Btn clicks and functions
function renderGauge(divId, minVal, maxVal, greenMax, yellowMin, yellowMax, redMin, redMax) {
try{
	var gauge = new Gauge({
		renderTo : divId,
		width : 200,
		height : 200,
		glow : true,
		units : '%',
		title : false,
		minValue : minVal,
		maxValue : maxVal,
		majorTicks : 0,
		minorTicks : 0,
		strokeTicks : false,
		highlights : [{
			from : 0,
			to : greenMax,
			color : 'lightgreen'
		}, {
			from : yellowMin,
			to : yellowMax,
			color : 'yellow'
		}, {
			from : redMin,
			to : redMax,
			color : 'red'
		}],
		colors : {
			plate : '#eeeeee',
			majorTicks : '#9e9e9e',
			minorTicks : '#eeeeee',
			title : '#000',
			units : '#9e9e9e',
			numbers : '#9e9e9e',
			needle : {
				start : 'blue',
				end : 'blue'
			}
		}
	});

	gauge.draw();

	return gauge;
	}
	catch(ex)
	{
		return null;
	}
}