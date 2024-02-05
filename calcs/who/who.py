import json
import os

dir = os.path.dirname(__file__)
json_path = os.path.join(dir,'tables','mena','mena.json')

with open(json_path) as json_file:
    who = json.load(json_file)



def WHO_Lab(gender = -1 , smoking_status = -1, age = -1, bp = -1,dm = -1 , tchol = -1   ):
    ages = ['70_74','65_69','60_64','55_59','50_54','45_49','40_44']
    bp_ranges = ['180_more','160_179','140_159','120_139','less_120']
    tchols = ['4', '4_49', '5_59', '6_69','7']


    if(tchol == -1 or age == -1 or smoking_status == -1 or bp == -1 or dm == -1):
        return 0
    
    rounded_bp = 0
    if(bp>=180):
        rounded_bp = 0
    elif(160 <= bp and bp <= 179):
        rounded_bp = 1
    elif(140 <= bp and bp <= 159):
        rounded_bp = 2
    elif(120 <= bp and bp <= 139):
        rounded_bp = 3
    elif(bp <= 119):
        rounded_bp =4
    else:
        return 0
    user_bp = bp_ranges[rounded_bp]
        
    rounded_age = 0
    if(age >= 70):
        rounded_age = 0
    elif(age >= 65 and age <= 69):
        rounded_age = 1
    elif(age >= 60 and age <= 64):
        rounded_age = 2
    elif(age >= 55 and age <= 59):
        rounded_age = 3
    elif(age >= 50 and age <= 54):
        rounded_age = 4
    elif(age >= 45 and age <= 49):
        rounded_age = 5
    elif(age <= 44):
        rounded_age = 6
    else:
        return 0
    user_age = ages[rounded_age]

    rounded_tchol = 0
    if(tchol < 4):
        rounded_tchol = 0
    elif( 4 <= tchol and tchol <= 4.9):
        rounded_tchol = 1
    elif( 5 <= tchol and tchol <= 5.9):
        rounded_tchol = 2
    elif( 6 <= tchol and tchol <= 6.9):
        rounded_tchol = 3
    elif(7 <= tchol):
        rounded_tchol = 4
    else:
        return 0
    
    user_tchol = tchols[rounded_tchol]

    if(dm == 0 ):
        user_dm_status = 'no_dm'
    elif(dm == 1):
        user_dm_status = 'dm'
    else:
        return 0

    if(gender == 0):
        user_gender = 'male'
    elif(gender == 1):
        user_gender = 'female'
    else:
        return 0
    
    if(smoking_status == 0 ):
        user_smoking_status = 'non_smoker'
    elif(smoking_status == 1):
        user_smoking_status = 'smoker'
    else:
        return 0
    
    return who['lab_based'][user_dm_status][user_gender][user_smoking_status][user_age][user_bp][user_tchol]

def WHO_No_Lab(gender = -1 , smoking_status = -1, age = -1, bp = -1, bmi = -1):
    ages = ['70_74','65_69','60_64','55_59','50_54','45_49','40_44']
    bp_ranges = ['180_more','160_179','140_159','120_139','less_120']
    bmis = ['less_20', '20_24', '25_29', '30_35','35_more']


    if(bmi == -1 or age == -1 or smoking_status == -1 or bp == -1 or bmi == -1 ):
        return 0
    
    rounded_bp = 0
    if(bp>=180):
        rounded_bp = 0
    elif(160 <= bp and bp <= 179):
        rounded_bp = 1
    elif(140 <= bp and bp <= 159):
        rounded_bp = 2
    elif(120 <= bp and bp <= 139):
        rounded_bp = 3
    elif(bp <= 119):
        rounded_bp =4
    else:
        return 0
    user_bp = bp_ranges[rounded_bp]
        
    rounded_age = 0
    if(age >= 70):
        rounded_age = 0
    elif(age >= 65 and age <= 69):
        rounded_age = 1
    elif(age >= 60 and age <= 64):
        rounded_age = 2
    elif(age >= 55 and age <= 59):
        rounded_age = 3
    elif(age >= 50 and age <= 54):
        rounded_age = 4
    elif(age >= 45 and age <= 49):
        rounded_age = 5
    elif(age <= 44):
        rounded_age = 6
    else:
        return 0
    user_age = ages[rounded_age]

    rounded_bmi = 0
    if(bmi < 20):
        rounded_bmi = 0
    elif( 20 <= bmi and bmi <= 24):
        rounded_bmi = 1
    elif( 25 <= bmi and bmi <= 29):
        rounded_bmi = 2
    elif( 30 <= bmi and bmi <= 35):
        rounded_bmi = 3
    elif(35 < bmi):
        rounded_bmi = 4
    else:
        return 0
    user_bmi = bmis[rounded_bmi]

    if(gender == 0):
        user_gender = 'male'
    elif(gender == 1):
        user_gender = 'female'
    else:
        return 0
    
    if(smoking_status == 0 ):
        user_smoking_status = 'non_somker'
    elif(smoking_status == 1):
        user_smoking_status = 'smoker'
    else:
        return 0
    
    return who['no_lab'][user_gender][user_smoking_status][user_age][user_bp][user_bmi]



