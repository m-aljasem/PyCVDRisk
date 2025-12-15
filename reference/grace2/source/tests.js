
test("Age in Decades",
    function () {
        equal(age_toDecades(0), 1);
        equal(age_toDecades(10), 1);
        equal(age_toDecades(15), 1.5);
        equal(age_toDecades(101), 10.1);  
    }
);

test("Age Xbhat1",
    function () {
        equal(age_calculateXbhat1(0), 0.000000000000);
        equal(age_calculateXbhat1(1), 0.411570000000);
        equal(age_calculateXbhat1(1.5), 0.617355000000);
        equal(age_calculateXbhat1(4.8), 1.975536000000);
        equal(age_calculateXbhat1(4.81644), 1.982302210800);
        equal(age_calculateXbhat1(4.8165), 1.982326905000);
        equal(age_calculateXbhat1(5.7), 2.345949000000);
        equal(age_calculateXbhat1(7.4), 3.045618000000);
        equal(age_calculateXbhat1(8.4), 3.457188000000);
        equal(age_calculateXbhat1(10.1), 4.156857000000);
    }
);

test("Age Xbhat1 Substituted",
    function () {
        equal(age_calculateXbhat1_substituded(0), 0.000000000000);
        equal(age_calculateXbhat1_substituded(1), 0.442300000000);
        equal(age_calculateXbhat1_substituded(1.5), 0.663450000000);
        equal(age_calculateXbhat1_substituded(4.8), 2.123040000000);
        equal(age_calculateXbhat1_substituded(4.81644), 2.130311412000);
        equal(age_calculateXbhat1_substituded(4.8165), 2.130337950000);
        equal(age_calculateXbhat1_substituded(5.7), 2.521110000000);
        equal(age_calculateXbhat1_substituded(7.4), 3.273020000000);
        equal(age_calculateXbhat1_substituded(8.4), 3.715320000000);
        equal(age_calculateXbhat1_substituded(10.1), 4.467230000000);
    }
);

test("Age f11",
    function () {
        equal(age_f11(0), 0.000000000000);
        equal(age_f11(4.8), 0.000000000000);
        equal(age_f11(4.81644), 0.000000000000);
        equal(age_f11(4.8165), 0.000000000000);
        equal(age_f11(5.7), 0.689635582875);
        equal(age_f11(7.4), 17.243499057875);
        equal(age_f11(8.4), 46.017415807875);
        equal(age_f11(9.0), 73.218245857875);
        equal(age_f11(10.1), 147.490869282875);
    }
);

test("Age f12",
    function () {
        equal(age_f12(0), 0.000000000000);
        equal(age_f12(4.8), 0.000000000000);
        equal(age_f12(4.81644), 0.000000000000);
        equal(age_f12(4.8165), 0.000000000000);
        equal(age_f12(5.7), 0.000000000000);
        equal(age_f12(7.4), 0.866711331009);
        equal(age_f12(8.4), 11.191822379417);
        equal(age_f12(9.0), 27.185958928579);
        equal(age_f12(10.1), 86.269369772823);
    }
);

test("Age f13",
    function () {
        equal(age_f13(0), 0.000000000000);
        equal(age_f13(4.8), 0.000000000000);
        equal(age_f13(4.81644), 0.000000000000);
        equal(age_f13(4.8165), 0.000000000000);
        equal(age_f13(5.7), 0.000000000000);
        equal(age_f13(7.4), 0.000000000000);
        equal(age_f13(8.4), 0.000825638562);
        equal(age_f13(9.0), 0.366720577724);
        equal(age_f13(10.1), 6.393920136968);
    }
);

test("Age Xbhat2",
    function () {
        equal(age_calculateXbhat2(0), 0.000000000000);
        equal(age_calculateXbhat2(4.8), 0.000000000000);
        equal(age_calculateXbhat2(4.81644), 0.000000000000);
        equal(age_calculateXbhat2(4.8165), 0.000000000000);
        equal(age_calculateXbhat2(5.7), 0.008896299019);
        equal(age_calculateXbhat2(7.4), 0.211260561677);
        equal(age_calculateXbhat2(8.4), 0.449260805965);
        equal(age_calculateXbhat2(10.1), 0.872238913447);
    }
);

test("Age Xbhat2 Substituted",
    function () {
        equal(age_calculateXbhat2_substituded(0), 0.000000000000);
        equal(age_calculateXbhat2_substituded(4.8), 0.000000000000);
        equal(age_calculateXbhat2_substituded(4.81644), 0.000000000000);
        equal(age_calculateXbhat2_substituded(4.8165), 0.000000000000);
        equal(age_calculateXbhat2_substituded(5.7), 0.008489414025);
        equal(age_calculateXbhat2_substituded(7.4), 0.201598256918);
        equal(age_calculateXbhat2_substituded(8.4), 0.428713218715);
        equal(age_calculateXbhat2_substituded(10.1), 0.832345815855);
    }
);

test("Age Xbhat",
    function () {
        equal(age_calculateXbhat(0), 0.000000000000);
        equal(age_calculateXbhat(48), 1.975536000000);
        equal(age_calculateXbhat(57), 2.354845299019);
        equal(age_calculateXbhat(74), 3.256878561677);
        equal(age_calculateXbhat(84), 3.906448805965);
        equal(age_calculateXbhat(101), 5.029095913447);
    }
);

test("Age Xbhat Substituted",
    function () {
        equal(age_calculateXbhat_substituted(0), 0.000000000000);
        equal(age_calculateXbhat_substituted(48), 2.123040000000);
        equal(age_calculateXbhat_substituted(57), 2.529599414025);
        equal(age_calculateXbhat_substituted(74), 3.474618256918);
        equal(age_calculateXbhat_substituted(84), 4.144033218715);
        equal(age_calculateXbhat_substituted(101), 5.299575815855);
    }
);

test("Pulse 30 beats per minute",
    function () {
        equal(pulse_to30BeatsPerMinute(0), 0.000000000000);
        equal(pulse_to30BeatsPerMinute(15), 0.500000000000);
        equal(pulse_to30BeatsPerMinute(25), 0.833333333333);
        equal(pulse_to30BeatsPerMinute(23), 0.766666666667);
        equal(pulse_to30BeatsPerMinute(28), 0.933333333333);
        equal(pulse_to30BeatsPerMinute(30), 1.000000000000);
        equal(pulse_to30BeatsPerMinute(40), 1.333333333333);
        equal(pulse_to30BeatsPerMinute(60), 2.000000000000);
        equal(pulse_to30BeatsPerMinute(85), 2.833333333333);
        equal(pulse_to30BeatsPerMinute(140), 4.666666666667);
        equal(pulse_to30BeatsPerMinute(200), 6.666666666667);
    }
);

test("Pulse Xbhat1",
    function () {
        equal(pulse_calculateXbhat1(0), 0.000000000000);
        equal(pulse_calculateXbhat1(0.5), 0.065690000000);
        equal(pulse_calculateXbhat1(0.833333333333), 0.109483333333);
        equal(pulse_calculateXbhat1(0.766666666667), 0.100724666667);
        equal(pulse_calculateXbhat1(0.933333333333), 0.122621333333);
        equal(pulse_calculateXbhat1(1), 0.131380000000);
        equal(pulse_calculateXbhat1(1.333333333333), 0.175173333333);
        equal(pulse_calculateXbhat1(2), 0.262760000000);
        equal(pulse_calculateXbhat1(2.833333333333), 0.372243333333);
        equal(pulse_calculateXbhat1(4.666666666667), 0.613106666667);
        equal(pulse_calculateXbhat1(6.666666666667), 0.875866666667);
    }
);

test("Pulse Xbhat1 Substituted",
    function () {
        equal(pulse_calculateXbhat1_substituted(0), 0.000000000000);
        equal(pulse_calculateXbhat1_substituted(0.5), 0.083525000000);
        equal(pulse_calculateXbhat1_substituted(0.833333333333), 0.139208333333);
        equal(pulse_calculateXbhat1_substituted(0.766666666667), 0.128071666667);
        equal(pulse_calculateXbhat1_substituted(0.933333333333), 0.155913333333);
        equal(pulse_calculateXbhat1_substituted(1), 0.167050000000);
        equal(pulse_calculateXbhat1_substituted(1.333333333333), 0.222733333333);
        equal(pulse_calculateXbhat1_substituted(2), 0.334100000000);
        equal(pulse_calculateXbhat1_substituted(2.833333333333), 0.473308333333);
        equal(pulse_calculateXbhat1_substituted(4.666666666667), 0.779566666667);
        equal(pulse_calculateXbhat1_substituted(6.666666666667), 1.113666666667);
    }
);

test("Pulse f31",
    function () {
        equal(pulse_f31(0), 0.000000000000);
        equal(pulse_f31(0.5), 0.000000000000);
        equal(pulse_f31(0.833333333333), 0.000000000000);
        equal(pulse_f31(0.766666666667), 0.000000000000);
        equal(pulse_f31(0.933333333333), 0.000000000000);
        equal(pulse_f31(1), 0.000000000000);
        equal(pulse_f31(1.333333333333), 0.000000000000);
        equal(pulse_f31(2), 0.024389000000);
        equal(pulse_f31(2.833333333333), 1.417509370369);
        equal(pulse_f31(4.666666666667), 25.846818629638);
        equal(pulse_f31(6.666666666667), 121.778085296321);
    }
);

test("Pulse f32",
    function () {
        equal(pulse_f32(0), 0.000000000000);
        equal(pulse_f32(0.5), 0.000000000000);
        equal(pulse_f32(0.833333333333), 0.000000000000);
        equal(pulse_f32(0.766666666667), 0.000000000000);
        equal(pulse_f32(0.933333333333), 0.000000000000);
        equal(pulse_f32(1), 0.000000000000);
        equal(pulse_f32(1.333333333333), 0.000000000000);
        equal(pulse_f32(2), 0.000000000000);
        equal(pulse_f32(2.833333333333), 0.000482239799);
        equal(pulse_f32(4.666666666667), 12.95206591840);
        equal(pulse_f32(6.666666666667), 112.317073828026);
    }
);

test("Pulse f33",
    function () {
        equal(pulse_f33(0), 0.000000000000);
        equal(pulse_f33(0.5), 0.000000000000);
        equal(pulse_f33(0.833333333333), 0.000000000000);
        equal(pulse_f33(0.766666666667), 0.000000000000);
        equal(pulse_f33(0.933333333333), 0.000000000000);
        equal(pulse_f33(1), 0.000000000000);
        equal(pulse_f33(1.333333333333), 0.000000000000);
        equal(pulse_f33(2), 0.000000000000);
        equal(pulse_f33(2.833333333333), 0.000000000000);
        equal(pulse_f33(4.666666666667), 0.330655288764);
        equal(pulse_f33(6.666666666667), 18.010796531708);
    }
);

test("Pulse Xbhat2",
    function () {
        equal(pulse_calculateXbhat2(0), 0.000000000000);
        equal(pulse_calculateXbhat2(0.5), 0.000000000000);
        equal(pulse_calculateXbhat2(0.833333333333), 0.000000000000);
        equal(pulse_calculateXbhat2(0.766666666667), 0.000000000000);
        equal(pulse_calculateXbhat2(0.933333333333), 0.000000000000);
        equal(pulse_calculateXbhat2(1), 0.000000000000);
        equal(pulse_calculateXbhat2(1.333333333333), 0.000000000000);
        equal(pulse_calculateXbhat2(2), 0.00979852464);
        equal(pulse_calculateXbhat2(2.833333333333), 0.569304819978);
        equal(pulse_calculateXbhat2(4.666666666667), 5.313439918081);
        equal(pulse_calculateXbhat2(6.666666666667), 11.037073582081);
    }
);

test("Pulse Xbhat2 Substituted",
    function () {
        equal(pulse_calculateXbhat2_substituted(0), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(0.5), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(0.833333333333), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(0.766666666667), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(0.933333333333), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(1), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(1.333333333333), 0.000000000000);
        equal(pulse_calculateXbhat2_substituted(2), 0.01071311214);
        equal(pulse_calculateXbhat2_substituted(2.833333333333), 0.622443337374);
        equal(pulse_calculateXbhat2_substituted(4.666666666667), 5.809392718081);
        equal(pulse_calculateXbhat2_substituted(6.666666666667), 12.067266382081);
    }
);

test("Pulse f34",
    function () {
        equal(pulse_f34(0), 0.000000000000);
        equal(pulse_f34(0.5), 0.000000000000);
        equal(pulse_f34(0.833333333333), 0.000000000000);
        equal(pulse_f34(0.766666666667), 0.000000000000);
        equal(pulse_f34(0.933333333333), 0.000000000000);
        equal(pulse_f34(1), 0.000000000000);
        equal(pulse_f34(1.333333333333), 0.000000000000);
        equal(pulse_f34(2), 0.000000000000);
        equal(pulse_f34(2.833333333333), 0.143329370370);
        equal(pulse_f34(4.666666666667), 13.088638629635);
        equal(pulse_f34(6.666666666667), 82.691905296315);
    }
);

test("Pulse f35",
    function () {
        equal(pulse_f35(0), 0.000000000000);
        equal(pulse_f35(0.5), 0.000000000000);
        equal(pulse_f35(0.833333333333), 0.000000000000);
        equal(pulse_f35(0.766666666667), 0.000000000000);
        equal(pulse_f35(0.933333333333), 0.000000000000);
        equal(pulse_f35(1), 0.000000000000);
        equal(pulse_f35(1.333333333333), 0.000000000000);
        equal(pulse_f35(2), 0.000000000000);
        equal(pulse_f35(2.833333333333), 0.000353068424);
        equal(pulse_f35(4.666666666667), 9.482762547400);
        equal(pulse_f35(6.666666666667), 82.232143338376);
    }
);

test("Pulse f36",
    function () {
        equal(pulse_f36(0), 0.000000000000);
        equal(pulse_f36(0.5), 0.000000000000);
        equal(pulse_f36(0.833333333333), 0.000000000000);
        equal(pulse_f36(0.766666666667), 0.000000000000);
        equal(pulse_f36(0.933333333333), 0.000000000000);
        equal(pulse_f36(1), 0.000000000000);
        equal(pulse_f36(1.333333333333), 0.000000000000);
        equal(pulse_f36(2), 0.000000000000);
        equal(pulse_f36(2.833333333333), 0.000000000000);
        equal(pulse_f36(4.666666666667), 0.143491917765);
        equal(pulse_f36(6.666666666667), 7.816006042062);
    }
);

test("Pulse Xbhat3",
    function () {
        equal(pulse_calculateXbhat3(0), 0.000000000000);
        equal(pulse_calculateXbhat3(0.5), 0.000000000000);
        equal(pulse_calculateXbhat3(0.833333333333), 0.000000000000);
        equal(pulse_calculateXbhat3(0.766666666667), 0.000000000000);
        equal(pulse_calculateXbhat3(0.933333333333), 0.000000000000);
        equal(pulse_calculateXbhat3(1), 0.000000000000);
        equal(pulse_calculateXbhat3(1.333333333333), 0.000000000000);
        equal(pulse_calculateXbhat3(2), 0.000000000000);
        equal(pulse_calculateXbhat3(2.833333333333), -0.196233544658);
        equal(pulse_calculateXbhat3(4.666666666667), -5.145970086320);
        equal(pulse_calculateXbhat3(6.666666666667), -11.358408822321);
    }
);

test("Pulse Xbhat3 Substituted",
    function () {
        equal(pulse_calculateXbhat3_substituted(0), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(0.5), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(0.833333333333), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(0.766666666667), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(0.933333333333), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(1), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(1.333333333333), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(2), 0.000000000000);
        equal(pulse_calculateXbhat3_substituted(2.833333333333), -0.211607786406);
        equal(pulse_calculateXbhat3_substituted(4.666666666667), -5.54913962736);
        equal(pulse_calculateXbhat3_substituted(6.666666666667), -12.248302155361);
    }
);

test("Pulse Xbhat",
    function () {
        equal(pulse_calculateXbhat(0), 0.000000000000);
        equal(pulse_calculateXbhat(15), 0.065690000000);
        equal(pulse_calculateXbhat(25), 0.109483333333);
        equal(pulse_calculateXbhat(23), 0.100724666667);
        equal(pulse_calculateXbhat(28), 0.122621333333);
        equal(pulse_calculateXbhat(30), 0.131380000000);
        equal(pulse_calculateXbhat(40), 0.175173333333);
        equal(pulse_calculateXbhat(60), 0.272558524640);
        equal(pulse_calculateXbhat(85), 0.745314608653);
        equal(pulse_calculateXbhat(140), 0.780576498428);
        equal(pulse_calculateXbhat(200), 0.554531426427);
    }
);

test("Pulse Xbhat Substituted",
    function () {
        equal(pulse_calculateXbhat_substituted(0), 0.000000000000);
        equal(pulse_calculateXbhat_substituted(15), 0.083525000000);
        equal(pulse_calculateXbhat_substituted(25), 0.139208333333);
        equal(pulse_calculateXbhat_substituted(23), 0.128071666667);
        equal(pulse_calculateXbhat_substituted(28), 0.155913333333);
        equal(pulse_calculateXbhat_substituted(30), 0.167050000000);
        equal(pulse_calculateXbhat_substituted(40), 0.222733333333);
        equal(pulse_calculateXbhat_substituted(60), 0.344813112140);
        equal(pulse_calculateXbhat_substituted(85), 0.884143884301);
        equal(pulse_calculateXbhat_substituted(140), 1.039819757388);
        equal(pulse_calculateXbhat_substituted(200), 0.932630893387);
    }
);

test("SBP Per 22 mmHg",
    function () {
        equal(sbp_toPerMinus22mmHg(0), 0.000000000000);
        equal(sbp_toPerMinus22mmHg(50), -2.500000000000);
        equal(sbp_toPerMinus22mmHg(101), -5.050000000000);
        equal(sbp_toPerMinus22mmHg(128), -6.400000000000);
        equal(sbp_toPerMinus22mmHg(145), -7.250000000000);
        equal(sbp_toPerMinus22mmHg(168), -8.400000000000);
        equal(sbp_toPerMinus22mmHg(200), -10.000000000000);
    }
);

test("SBP Xbhat1",
    function () {
        equal(sbp_calculateXbhat1(0.000000000000), 0.000000000000);
        equal(sbp_calculateXbhat1(-2.500000000000), -0.205550000000);
        equal(sbp_calculateXbhat1(-5.050000000000), -0.415211000000);
        equal(sbp_calculateXbhat1(-6.400000000000), -0.526208000000);
        equal(sbp_calculateXbhat1(-7.250000000000), -0.596095000000);
        equal(sbp_calculateXbhat1(-8.400000000000), -0.690648000000);
        equal(sbp_calculateXbhat1(-10.000000000000), -0.822200000000);
    }
);

test("SBP Xbhat1 Substituted",
    function () {
        equal(sbp_calculateXbhat1_substituted(0.000000000000), 0.000000000000);
        equal(sbp_calculateXbhat1_substituted(-2.500000000000), -0.058450000000);
        equal(sbp_calculateXbhat1_substituted(-5.050000000000), -0.118069000000);
        equal(sbp_calculateXbhat1_substituted(-6.400000000000), -0.149632000000);
        equal(sbp_calculateXbhat1_substituted(-7.250000000000), -0.169505000000);
        equal(sbp_calculateXbhat1_substituted(-8.400000000000), -0.196392000000);
        equal(sbp_calculateXbhat1_substituted(-10.000000000000), -0.233800000000);
    }
);

test("SBP f21",
    function () {
        equal(sbp_f21(0.000000000000), 733.870808000000);
        equal(sbp_f21(-2.500000000000), 277.167808000000);
        equal(sbp_f21(-5.050000000000), 62.570773000000);
        equal(sbp_f21(-6.400000000000), 17.984728000000);
        equal(sbp_f21(-7.250000000000), 5.545233000000);
        equal(sbp_f21(-8.400000000000), 0.238328000000);
        equal(sbp_f21(-10.000000000000), 0.000000000000);
    }
);

test("SBP f22",
    function () {
        equal(sbp_f22(0.000000000000), 759.304745515152);
        equal(sbp_f22(-2.500000000000), 200.281275818182);
        equal(sbp_f22(-5.050000000000), 15.871627636364);
        equal(sbp_f22(-6.400000000000), 0.415281272727);
        equal(sbp_f22(-7.250000000000), 0.000000000000);
        equal(sbp_f22(-8.400000000000), 0.000000000000);
        equal(sbp_f22(-10.000000000000), 0.000000000000);
    }
);

test("SBP f23",
    function () {
        equal(sbp_f23(0.000000000000), 187.070287515152);
        equal(sbp_f23(-2.500000000000), 27.862317818182);
        equal(sbp_f23(-5.050000000000), 0.024454636364);
        equal(sbp_f23(-6.400000000000), 0.000000000000);
        equal(sbp_f23(-7.250000000000), 0.000000000000);
        equal(sbp_f23(-8.400000000000), 0.000000000000);
        equal(sbp_f23(-10.000000000000), 0.000000000000);
    }
);

test("SBP Xbhat2",
    function () {
        equal(sbp_calculateXbhat2(0.000000000000), 1.648690770000);
        equal(sbp_calculateXbhat2(-2.500000000000), 1.068438270000);
        equal(sbp_calculateXbhat2(-5.050000000000), 0.476580720000);
        equal(sbp_calculateXbhat2(-6.400000000000), 0.179208356618);
        equal(sbp_calculateXbhat2(-7.250000000000), 0.056561376600);
        equal(sbp_calculateXbhat2(-8.400000000000), 0.002430945600);
        equal(sbp_calculateXbhat2(-10.000000000000), 0.000000000000);
    }
);

test("SBP Xbhat2 Substituted",
    function () {
        equal(sbp_calculateXbhat2_substituted(0.000000000000), 2.9660270225);
        equal(sbp_calculateXbhat2_substituted(-2.500000000000), 1.922141397500);
        equal(sbp_calculateXbhat2_substituted(-5.050000000000), 0.857378060000);
        equal(sbp_calculateXbhat2_substituted(-6.400000000000), 0.322399347445);
        equal(sbp_calculateXbhat2_substituted(-7.250000000000), 0.10175502555);
        equal(sbp_calculateXbhat2_substituted(-8.400000000000), 0.0043733188000);
        equal(sbp_calculateXbhat2_substituted(-10.000000000000), 0.000000000000);
    }
);

test("SBP Xbhat",
    function () {
        equal(sbp_calculateXbhat(0), 1.648690770000);
        equal(sbp_calculateXbhat(50), 0.862888270000);
        equal(sbp_calculateXbhat(101), 0.061369720000);
        equal(sbp_calculateXbhat(128), -0.346999643382);
        equal(sbp_calculateXbhat(145), -0.539533623400);
        equal(sbp_calculateXbhat(168), -0.688217054400);
        equal(sbp_calculateXbhat(200), -0.822200000000);
    }
);

test("SBP Xbhat Substituted",
    function () {
        equal(sbp_calculateXbhat_substituted(0), 2.966027022500);
        equal(sbp_calculateXbhat_substituted(50), 1.863691397500);
        equal(sbp_calculateXbhat_substituted(101), 0.739309060000);
        equal(sbp_calculateXbhat_substituted(128), 0.172767347445);
        equal(sbp_calculateXbhat_substituted(145), -0.067749974450);
        equal(sbp_calculateXbhat_substituted(168), -0.192018681200);
        equal(sbp_calculateXbhat_substituted(200), -0.233800000000);
    }
);

test("Creatine Xbhat1",
    function () {
        equal(creatine_calculateXbhat1(0), 0.000000000000);
        equal(creatine_calculateXbhat1(0.5), -0.256295000000);
        equal(creatine_calculateXbhat1(0.9), -0.461331000000);
        equal(creatine_calculateXbhat1(1), -0.512590000000);
        equal(creatine_calculateXbhat1(1.3), -0.666367000000);
        equal(creatine_calculateXbhat1(2), -1.025180000000);
        equal(creatine_calculateXbhat1(2.5), -1.281475000000);
    }
);

test("Creatine f41",
    function () {
        equal(creatine_f41(0), 0.000000000000);
        equal(creatine_f41(0.5), 0.000000000000);
        equal(creatine_f41(0.9), 0.009938375000);
        equal(creatine_f41(1), 0.031255875000);
        equal(creatine_f41(1.3), 0.232608375000);
        equal(creatine_f41(2), 2.273930875000);
        equal(creatine_f41(2.5), 5.979018375000);
    }
);

test("Creatine f42",
    function () {
        equal(creatine_f42(0), 0.000000000000);
        equal(creatine_f42(0.5), 0.000000000000);
        equal(creatine_f42(0.9), 0.000000000000);
        equal(creatine_f42(1), 0.000000000000);
        equal(creatine_f42(1.3), 0.005585812500);
        equal(creatine_f42(2), 0.937539562500);
        equal(creatine_f42(2.5), 3.731720812500);
    }
);

test("Creatine f43",
    function () {
        equal(creatine_f43(0), 0.000000000000);
        equal(creatine_f43(0.5), 0.000000000000);
        equal(creatine_f43(0.9), 0.000000000000);
        equal(creatine_f43(1), 0.000000000000);
        equal(creatine_f43(1.3), 0.000000000000);
        equal(creatine_f43(2), 0.000000000000);
        equal(creatine_f43(2.5), 0.041156437500);
    }
);

test("Creatine Xbhat2",
    function () {
        equal(creatine_calculateXbhat2(0), 0.000000000000);
        equal(creatine_calculateXbhat2(0.5), 0.000000000000);
        equal(creatine_calculateXbhat2(0.9), 0.074799589298);
        equal(creatine_calculateXbhat2(1), 0.235242342248);
        equal(creatine_calculateXbhat2(1.3), 1.708648993046);
        equal(creatine_calculateXbhat2(2), 10.058135390921);
        equal(creatine_calculateXbhat2(2.5), 17.223682878360);
    }
);

test("Creatine f44",
    function () {
        equal(creatine_f44(0), 0.000000000000);
        equal(creatine_f44(0.5), 0.000000000000);
        equal(creatine_f44(0.9), 0.000000000000);
        equal(creatine_f44(1), 0.000421875000);
        equal(creatine_f44(1.3), 0.052734375000);
        equal(creatine_f44(2), 1.242296875000);
        equal(creatine_f44(2.5), 3.906984375000);
    }
);

test("Creatine f45",
    function () {
        equal(creatine_f45(0), 0.000000000000);
        equal(creatine_f45(0.5), 0.000000000000);
        equal(creatine_f45(0.9), 0.000000000000);
        equal(creatine_f45(1), 0.000000000000);
        equal(creatine_f45(1.3), 0.004614366848);
        equal(creatine_f45(2), 0.774489203804);
        equal(creatine_f45(2.5), 3.082725888587);
    }
);

test("Creatine f46",
    function () {
        equal(creatine_f46(0), 0.000000000000);
        equal(creatine_f46(0.5), 0.000000000000);
        equal(creatine_f46(0.9), 0.000000000000);
        equal(creatine_f46(1), 0.000000000000);
        equal(creatine_f46(1.3), 0.000000000000);
        equal(creatine_f46(2), 0.000000000000);
        equal(creatine_f46(2.5), 0.019683513587);
    }
);

test("Creatine Xbhat3",
    function () {
        equal(creatine_calculateXbhat3(0), 0.000000000000);
        equal(creatine_calculateXbhat3(0.5), 0.000000000000);
        equal(creatine_calculateXbhat3(0.9), 0.000000000000);
        equal(creatine_calculateXbhat3(1), -0.007690878281);
        equal(creatine_calculateXbhat3(1.3), -0.877238816213);
        equal(creatine_calculateXbhat3(2), -8.528241441667);
        equal(creatine_calculateXbhat3(2.5), -15.385256766660);
    }
);

test("Creatine Xbhat",
    function () {
        equal(creatine_calculateXbhat(0), 0.000000000000);
        equal(creatine_calculateXbhat(0.5), -0.256295000000);
        equal(creatine_calculateXbhat(0.9), -0.386531410702);
        equal(creatine_calculateXbhat(1), -0.285038536033);
        equal(creatine_calculateXbhat(1.3), 0.165043176833);
        equal(creatine_calculateXbhat(2), 0.504713949254);
        equal(creatine_calculateXbhat(2.5), 0.556951111700);
    }
);

test("Creatine Xbhat Substituted",
    function () {
        equal(creatine_calculateXbhat_substituted(0), 0);
        equal(creatine_calculateXbhat_substituted(true), 0.44469);
        equal(creatine_calculateXbhat_substituted(false), 0);
        equal(creatine_calculateXbhat_substituted(), 0);
    }
);

test("Killip Xbhat",
    function () {
        equal(killip_calculateXbhat(), undefined);
        equal(killip_calculateXbhat(null), undefined);
        equal(killip_calculateXbhat(0), undefined);
        equal(killip_calculateXbhat(1), 0);
        equal(killip_calculateXbhat(2), 0.63827);
        equal(killip_calculateXbhat(3), 0.85325);
        equal(killip_calculateXbhat(4), 1.29372);
        equal(killip_calculateXbhat(5), undefined);
    }
);

test("Killip Xbhat Substituted",
    function () {
        equal(killip_calculateXbhat_substituted(), 0);
        equal(killip_calculateXbhat_substituted(1), 0.68008);
        equal(killip_calculateXbhat_substituted(true), 0.68008);
        equal(killip_calculateXbhat_substituted(false), 0);
    }
);

test("Cardiac Xbhat",
    function () {
        equal(cardiac_calculateXbhat(), 0);
        equal(cardiac_calculateXbhat(null), 0);
        equal(cardiac_calculateXbhat(0), 0);
        equal(cardiac_calculateXbhat(1), 0.87185);
        equal(cardiac_calculateXbhat(true), 0.87185);
        equal(cardiac_calculateXbhat('true'), 0.87185);
        equal(cardiac_calculateXbhat('false'), 0.87185);
        equal(cardiac_calculateXbhat(false), 0);
        equal(cardiac_calculateXbhat(3), 0.87185);
        equal(cardiac_calculateXbhat(4), 0.87185);
        equal(cardiac_calculateXbhat(5), 0.87185);
    }
);

test("Cardiac Xbhat Substituted",
    function () {
        equal(cardiac_calculateXbhat_substituted(), 0);
        equal(cardiac_calculateXbhat_substituted(null), 0);
        equal(cardiac_calculateXbhat_substituted(0), 0);
        equal(cardiac_calculateXbhat_substituted(1), 1.18193);
        equal(cardiac_calculateXbhat_substituted(true), 1.18193);
        equal(cardiac_calculateXbhat_substituted(false), 0);
        equal(cardiac_calculateXbhat_substituted(3), 1.18193);
        equal(cardiac_calculateXbhat_substituted(4), 1.18193);
        equal(cardiac_calculateXbhat_substituted(5), 1.18193);
    }
);

test("Enzymes Xbhat",
    function () {
        equal(enzymes_calculateXbhat(), 0);
        equal(enzymes_calculateXbhat(null), 0);
        equal(enzymes_calculateXbhat(0), 0);
        equal(enzymes_calculateXbhat(1), 0.37660);
        equal(enzymes_calculateXbhat(true), 0.37660);
        equal(enzymes_calculateXbhat(false), 0);
        equal(enzymes_calculateXbhat(3), 0.37660);
        equal(enzymes_calculateXbhat(4), 0.37660);
        equal(enzymes_calculateXbhat(5), 0.37660);
    }
);

test("Enzymes Xbhat Substituted",
    function () {
        equal(enzymes_calculateXbhat_substituted(), 0);
        equal(enzymes_calculateXbhat_substituted(null), 0);
        equal(enzymes_calculateXbhat_substituted(0), 0);
        equal(enzymes_calculateXbhat_substituted(1), 0.39944);
        equal(enzymes_calculateXbhat_substituted(true), 0.39944);
        equal(enzymes_calculateXbhat_substituted(false), 0);
        equal(enzymes_calculateXbhat_substituted(3), 0.39944);
        equal(enzymes_calculateXbhat_substituted(4), 0.39944);
        equal(enzymes_calculateXbhat_substituted(5), 0.39944);
    }
);

test("STEMI Xbhat",
    function () {
        equal(stemi_calculateXbhat(), 0);
        equal(stemi_calculateXbhat(null), 0);
        equal(stemi_calculateXbhat(0), 0);
        equal(stemi_calculateXbhat(1), 0.44303);
        equal(stemi_calculateXbhat(true), 0.44303);
        equal(stemi_calculateXbhat(false), 0);
        equal(stemi_calculateXbhat(3), 0.44303);
        equal(stemi_calculateXbhat(4), 0.44303);
        equal(stemi_calculateXbhat(5), 0.44303);
    }
);

test("STEMI Xbhat Substituted",
    function () {
        equal(stemi_calculateXbhat_substituted(), 0);
        equal(stemi_calculateXbhat_substituted(null), 0);
        equal(stemi_calculateXbhat_substituted(0), 0);
        equal(stemi_calculateXbhat_substituted(1), 0.46524);
        equal(stemi_calculateXbhat_substituted(true), 0.46524);
        equal(stemi_calculateXbhat_substituted(false), 0);
        equal(stemi_calculateXbhat_substituted(3), 0.46524);
        equal(stemi_calculateXbhat_substituted(4), 0.46524);
        equal(stemi_calculateXbhat_substituted(5), 0.46524);
    }
);

test("One year long form",
    function () {
        equal(oneyear_longform(90, 92, 139, 1.06, false, true, true, 1), 0.28);
        equal(oneyear_longform(40, 82, 107, 1.00, false, false, true, 1), 0.02);
    }
);

    test("One year substituted",
    function () {
        equal(oneyear_substituted(90, 92, 139, false, false, true, true, false), 0.30);
        equal(oneyear_substituted(40, 82, 107, false, false, false, true, false), 0.02);
    }
);