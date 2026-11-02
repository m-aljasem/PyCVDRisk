## PREDICT CVD Risk Scores

This is an R package featuring functions for generating PREDICT cardiovascular disease (CVD) risk scores.

### Features
-	Estimate 5 year CVD risk in people with and without established CVD
-   Estimate 5 year CVD risk in people who have experienced an acute coronary syndrome (ACS)
-   Estimate 5 year CVD risk in people without established Type-II diabetes
-   Estimate 5 year major bleeding risk in people without established CVD

### Functions
1. `NoPriorCVDRisk` 
    calculates the 5 year absolute risk of CVD for people without a history of atherosclerotic CVD. Published in the Lancet (Pylypchuk et al. 2018) and does not include BMI as a predictor. 
    [Full Article](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(18)30664-0/fulltext)
    
2. `NoPriorCVDRisk_BMI` 
    calculates the 5 year absolute risk of CVD for people without a history of atherosclerotic CVD. Published as a Ministry of Health's HISO standard and adds BMI as a predictor to the Lancet equation.
    [HISO Standard](https://www.health.govt.nz/publication/hiso-100712019-cardiovascular-disease-risk-assessment-data-standard)
    
3. `NoPriorCVDRisk_Policy` 
    calculates the 5 year absolute risk of CVD for people without a history of atherosclerotic CVD that is intended for use at policy or general population level. Publushed in the 
    IJE (Mehta et al. 2018) and uses routinely-collected health information as predictors.
    [Full Article](https://academic.oup.com/ije/article/47/5/1571/5053287)
    
4. `NoPriorCVDBleedRisk` 
    calculates the 5 year absolute risk of major bleeding for people without a history of atherosclerotic CVD. Published in AnnIntMed (Selak et al. 2019).
    [Full Article](https://www.annals.org/aim/fullarticle/doi/10.7326/M18-2808)
    
5. `NoPriorT2DRisk` 
    creates a 5 year absolute risk of CVD for people without a history of type-II diabetes. Published as a Ministry of Health's HISO standard and includes BMI as a predictor.
    [HISO Standard](https://www.health.govt.nz/publication/hiso-100712019-cardiovascular-disease-risk-assessment-data-standard)
    
6. `PostCVDRisk` 
    creates a 5 year absolute risk of CVD for people with prior CVD. This equation is yet to be published. However, it is an update to the previously published equation in Heart (Poppe et al. 2017).
    [Full Article](https://heart.bmj.com/content/103/12/891.1)
    
7. `PostACSRisk` 
    creates a 5 year absolute risk for people for people who have experienced an ACS event. Published in Heart (Poppe et al. 2019).
    [Full Article](https://heart.bmj.com/content/early/2019/12/10/heartjnl-2019-315809.full)

### Installation
To install package:
```r
remotes::install_github("VIEW2020/PredictRiskScores")
```

To load the package:
```r
library(PredictRiskScores)
```
### Usage
#### Exclusions
<a href="https://github.com/VIEW2020/PredictRiskScores/wiki" target="_blank">See Wiki</a> for details regarding exclusion critiera as each function is only applicable to specific populations. For example, primary prevention equations for estimating CVD risk should be implemented on a population without CVD, vis-à-vis `NoPriorCVDRisk`, `NoPriorCVDRisk_BMI`, `NoPriorCVDRisk_Policy`, and `NoPriorCVDBleedRisk`. Further, CVD risk can be estimated for people without both CVD and diabetes using `NoPriorT2DRisk`. For secondary prevention, estimating subsequent CVD risk should be implmented on a population with prior CVD, vis-à-vis `PostCVDRisk` and `PostACSRisk`. 


#### One-off Calculator
Each function can be used as either a calculator or vectoriser. When used as a one-off calculator, a dataset is not required. This can be handy when checking the risk estimate for 
an individual person or for quickly seeing the effect of changing a single parameter value. 

```r
PostACSRisk(sex="F", age=65, eth=Indian, nzdep=5, smoker=0, diabetes=0,
            af=0, hf=1, acsdays=65, acstype="NSTEMI", bmi=NA, sbp=118,
            tchdl=3.3, hba1c=NA, scr=52, bpl=1, lld=1, athrombi=1)
```
```
[1] 0.4877
```
All parameter values can be numeric or encoded. Values for some parameters such as BMI and HbA1C can be `NA` if the value is unknown. See *values* in R documentation. Binary variables can take `TRUE`/`FALSE` boolean values.
```r
PostACSRisk(sex=0, age=65, eth=43, nzdep=5, smoker=0, diabetes=0,
            af=0, hf=1, acsdays=65, acstype=1, bmi=NA, sbp=118,
            tchdl=3.3, hba1c=NA, scr=52, bpl=T, lld=F, athrombi=1)
```
```
[1] 0.4877
```
Binary parameters can take a variety of input values. For example, the value for `TRUE` can be `T`, `Y`, `Yes`, or `1`, while the value for `FALSE` can be `F`, `N`, `No`, or `0`. Quotations are not required and case sensitivity is ignored.  
```r
NoPriorCVDRisk(sex=F, age=30, eth=M, exsmoker=0, smoker=TRUE, nzdep=3, diabetes=Y,
               af=F, familyhx=1, lld=y, athrombi=yes, bpl=T, sbp=150, tchdl=5)

```
```
[1] 0.0549
```

#### Vectorise Dataset
When a dataset is supplied, a risk score is produced for each row of data. Each argument requires the variable name from the dataset.
This can be handy when risk estimates are required for each row of data, or when datasets require vectorisation.

```r
NoPriorCVDRisk(dat=DATA, sex=sex, age=age, eth=ethnicity, exsmoker=exsmoker, smoker=current_smk, nzdep=nzdep, 
              diabetes=diab_status, af=hx_af, familyhx=familyhx, lld=lld, athrombi=athrombotics, bpl=bpl, 
              sbp=sbp, tchdl=tchdl)
```
```
[1] 0.3327 0.0489 0.0620 0.5441 0.0688 0.1672 0.5054 0.0442 0.0150 0.0387
```
When a dataset is supplied, the function returns a numeric vector of risk scores. Each element of the vector is positioned as per row index. As such, the resulting numeric vector can be 
assigned back to the dataset as a new variable.
```r
DATA$riskscores <- NoPriorCVDRisk(dat=DATA, sex=sex, age=age, eth=ethnicity, exsmoker=exsmoker, smoker=current_smk,
                                  nzdep=nzdep, diabetes=diab_status, af=hx_af, familyhx=familyhx, lld=lld, 
                                  athrombi=athrombotics, bpl=bpl, sbp=sbp, tchdl=tchdl)l)
```

### Further Arguements
Each function contains 3 dot parameters: `dp`, `allow.age`, and `allow.na`. These arguments can be specified to change their default settings.

- **Decimal Places** by default will return risk scores to 4 decimal places. The number of decimal places can be changed using the argument `dp`. For example, `dp = 6` will produce risk scores containing 6 decimal places. <br><br>
- **Age Allowance** by default will return risk scores for those between 18 and 110 years. Any age inputs outside of this range will return and `NA`. Ages 18 - 29 are calculated as 30; and 80 - 110 as 79. If `allow.age` is set to `FALSE`, then age values between 30 and 74 must be provided, else `NA` is returned as the risk score. **Note!** For equations involving prior CVD (i.e.  `PostCVDRisk` and `PostACSRisk`), age values between 30 and 79 must be provided if `allow.age` is set to `FALSE`, else `NA` is returned as the risk score.  <br><br>
- **Missing Allowance** by default will return risk scores where the missing values are treated as `0`, `No`, or `FALSE`. This applies only to binary variables including smoking status. If `allow.na` is set to `FALSE`, then binary inputs including smoking status must have a value, else `NA` will be returned as the risk score.


### Integration with `data.table` and `dplyr`
The suite of functions in this package can be integrated into both `data.table` and `dplyr`. For example, when datasets are extremely large, consider 
using `data.table` along with the `:=` notation. In the example below, a new column called `riskscore` is created.
The `data.table` syntax might seem confusing at first but it offers fast and efficient performance.

```r
library(data.table); setDT(DATA)

DATA[, riskscore := NoPriorCVDRisk(dat=DATA, sex=sex, age=age, eth=ethnicity, exsmoker=exsmoker, smoker=current_smk,
                                  nzdep=nzdep, diabetes=diab_status, af=hx_af, familyhx=familyhx, lld=lld,
                                  athrombi=athrombotics, bpl=bpl, sbp=sbp, tchdl=tchdl)]
```
Users who are more comfortable with `dplyr` will find that the functions works with `mutate`. In the example below, a new column called `riskscore` is created.
```r
library(dplyr)

DATA %>%
  mutate(riskscore = NoPriorCVDRisk(dat=DATA, sex=sex, age=age, eth=ethnicity, exsmoker=exsmoker, smoker=current_smk,
                                    nzdep=nzdep, diabetes=diab_status, af=hx_af, familyhx=familyhx, lld=lld,
                                    athrombi=athrombotics, bpl=bpl, sbp=sbp, tchdl=tchdl))
```

#### R Documentation
```r
library(PredictRiskScores)

?NoPriorCVDRisk
?NoPriorCVDRisk_BMI
?NoPriorCVDRisk_Policy
?NoPriorCVDBleedRisk
?NoPriorT2DRisk
?PostCVDRisk
?PostACSRisk
```

### Maintainer / Developer 
- Billy Wu / billy.wu@auckland.ac.nz
### Contributor / Tester 
- Catherine Choi / yeunhyang.choi@auckland.ac.nz


### See Also
I built a web-based risk calculator using the bleeding risk equation developed by Selak et al (2019) and the general CVD risk equation developed by Pylypchuk et al (2018), to provides clinicians with an individualised estimate of the CVD benefit and bleeding harms of aspirin for their patients without established CVD. [Full Article](https://annals.org/aim/fullarticle/2751452/personalized-prediction-cardiovascular-benefits-bleeding-harms-from-aspirin-primary-prevention).

<a href="https://aspirinbenefitharmcalculator.shinyapps.io/calculator/" target="_blank">Link to the risk calculator here</a>

### Coming Soon
-	Estimate 5 year CVD risk in people with severe mental illness
-   Estimate 5 year diabetes risk in people without established diabetes
-   Estimate 1 year acute CVD risk in people with established ACS
-   Estimate 1 year acute bleeding risk in people with established ACS

