from calcs.score2.score2 import SCORE2
from calcs.who.who import WHO_No_Lab, WHO_Lab


print(SCORE2(age = 58, bp = 159, tchol = 10, hdl = 5, gender = 1, smoking= 1))
print(WHO_No_Lab(gender = 1 , smoking_status = 1, age = 45, bp = 145, bmi = 30))

print(WHO_Lab(gender = 1 , smoking_status = 0, age = 66, bp = 130,dm = 1 , tchol = 5))
