/* Runs ClinRisk's QRISK3-2017 functions (source/qrisk3_2017.c, unmodified) on CSV rows.
 * Build and run: see run_reference.sh. Input columns (no header):
 * sex(1=male),age,AF,atypicalantipsy,corticosteroids,impotence2,migraine,ra,renal,semi,sle,
 * treatedhyp,type1,type2,bmi,ethrisk,fh_cvd,rati,sbp,sbps5,smoke_cat,town
 * Output: the score (percent) printed to 17 significant digits. */
#include <math.h>
#include <stdio.h>
#include "source/qrisk3_2017.c"

int main(void) {
	int male, age, af, aty, cor, imp, mig, ra, ren, semi, sle, trt, t1, t2, eth, fh, smoke;
	double bmi, rati, sbp, sbps5, town;
	while (scanf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lf,%d,%d,%lf,%lf,%lf,%d,%lf",
	             &male, &age, &af, &aty, &cor, &imp, &mig, &ra, &ren, &semi, &sle, &trt, &t1, &t2,
	             &bmi, &eth, &fh, &rati, &sbp, &sbps5, &smoke, &town) == 22) {
		double s = male
			? cvd_male_raw(age, af, aty, cor, imp, mig, ra, ren, semi, sle, trt, t1, t2, bmi, eth, fh, rati, sbp, sbps5, smoke, 10, town)
			: cvd_female_raw(age, af, aty, cor, mig, ra, ren, semi, sle, trt, t1, t2, bmi, eth, fh, rati, sbp, sbps5, smoke, 10, town);
		printf("%.17g\n", s);
	}
	return 0;
}
