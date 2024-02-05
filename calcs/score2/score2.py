import json
import os

dir = os.path.dirname(__file__)
json_path = os.path.join(dir,'SCORE2-HR.json')

with open(json_path) as json_file:
    score2 = json.load(json_file)


def SCORE2(age = 0 , bp =0 , gender = 0, tchol = 0, hdl=0, smoking = 0):
    ages = ['65_69', '60_64', '55_59', '50_54', '45_49', '40_44']
    bps = ['160_179', '140_159', '120_139', '100_119']
    genders = ['male', 'female']
    smoking_status = ['non_smoking', 'smoking']
    chol = ['30_39', '40_49', '50_59', '60_69']
    result = 0
    
    round_bp = 0
    round_chol = 0
    round_gender = 0
    round_smoking_status = 0
    round_age = 0
    
    #BP Round
    if bp == 0:
        result = 0
    elif bp < 120:
        round_bp = bps[3]
    elif ((bp >= 120) and (bp < 140)):
        round_bp = bps[2]
    elif ((bp >= 140) and (bp < 160)):
        round_bp = bps[1]
    elif bp >= 160:
        round_bp = bps[0]
    
    #Age Round
    if age == 0:
        result = 0
    elif age < 45:
        round_age = ages[5]
    elif ((age >= 45) and (age < 50)):
        round_age = ages[4]
    elif ((age >= 50) and (age < 55)):
        round_age = ages[3]
    elif ((age >= 55) and (age < 60)):
        round_age = ages[2]
    elif ((age >= 60) and (age < 65)):
        round_age = ages[1]
    elif age >= 65:
        round_age = ages[0]
    
    #Chol Round
    non_hdl_chol = tchol - hdl
    if non_hdl_chol == 0:
        result = 0
    elif non_hdl_chol < 4:
        round_chol = chol[0]
    elif ((non_hdl_chol >= 4) and (non_hdl_chol < 5)):
        round_chol = chol[1]
    elif ((non_hdl_chol >= 5) and (non_hdl_chol < 6)):
        round_chol = chol[2]
    elif non_hdl_chol >= 6:
        round_chol = chol[3]

    #Gender
    round_gender = genders[gender]

    #Smoking Status
    round_smoking_status = smoking_status[smoking]


    return score2[round_gender][round_smoking_status][round_age][round_bp][round_chol]
    
    
