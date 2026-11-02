/* Runs ClinRisk's public QRISK2-2014 functions cvd_female/cvd_male (source/c, unmodified,
 * including their input validation) on CSV rows. Input columns (no header):
 * sex(1=male),age,AF,ra,renal,treatedhyp,type1,type2,bmi,ethrisk,fh_cvd,rati,sbp,smoke_cat,town
 * Output: the score (percent) to 17 significant digits, or "nan" when ClinRisk reports an error. */
#include <stdio.h>
#include <clinrisk/Q80_model_4_0.h>
#include <clinrisk/Q80_model_4_1.h>

int main(void) {
	int male, age, af, ra, ren, trt, t1, t2, eth, fh, smoke, err;
	double bmi, rati, sbp, town, s;
	char buf[1024];
	while (scanf("%d,%d,%d,%d,%d,%d,%d,%d,%lf,%d,%d,%lf,%lf,%d,%lf",
	             &male, &age, &af, &ra, &ren, &trt, &t1, &t2, &bmi, &eth, &fh, &rati, &sbp, &smoke, &town) == 15) {
		s = male ? cvd_male(age, af, ra, ren, trt, t1, t2, bmi, eth, fh, rati, sbp, smoke, 10, town, &err, buf, sizeof buf)
		         : cvd_female(age, af, ra, ren, trt, t1, t2, bmi, eth, fh, rati, sbp, smoke, 10, town, &err, buf, sizeof buf);
		if (err) printf("nan\n"); else printf("%.17g\n", s);
	}
	return 0;
}
