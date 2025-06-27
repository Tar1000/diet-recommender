from sklearn.preprocessing import LabelEncoder

def classify_sugar_level(fbs):
    if fbs < 100:
        return "Controlled"
    elif 100 <= fbs < 126:
        return "Moderate"
    else:
        return "High"

def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

sugar_encoder = LabelEncoder()
bmi_encoder = LabelEncoder()
diet_encoder = LabelEncoder()

def encode_dataset(df):
    df = df.copy()
    df['Sugar_Level_Code'] = sugar_encoder.fit_transform(df['Sugar_Level_Category'])
    df['BMI_Code'] = bmi_encoder.fit_transform(df['BMI_Category'])
    df['Diet_Code'] = diet_encoder.fit_transform(df['Diet_Type'])
    return df

def encode_user_input(sugar_level, bmi_category, diet_type):
    return [
        sugar_encoder.transform([sugar_level])[0],
        bmi_encoder.transform([bmi_category])[0],
        diet_encoder.transform([diet_type])[0]
    ]
