from cvdrisk import SCORE2,WHO_Lab,WHO_No_Lab,SCORE2OP

print(SCORE2(age = 58, bp = 159, tchol = 10, hdl = 5, gender = 1, smoking= 1))
print(WHO_No_Lab(gender = 1 , smoking_status = 1, age = 45, bp = 145, bmi = 30))
print(WHO_Lab(gender = 1 , smoking_status = 0, age = 66, bp = 130,dm = 1 , tchol = 5))
print(SCORE2OP(age = 77 , bp =150 , gender = 1, tchol = 8, hdl=2, smoking = 1))