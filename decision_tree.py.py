from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder


# Step 2: Build the Decision Tree Model
def build_decision_tree_model(features, labels):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(features, labels)
    return model

# Example data (symptoms and priorities)
symptoms_data = [
    {'symptom': 'Fever', 'priority': 1},
    {'symptom': 'Cough', 'priority': 1},
    {'symptom': 'Headache', 'priority': 1},
    {'symptom': 'Fatigue', 'priority': 1},
    {'symptom': 'Sore Throat', 'priority': 1},
    {'symptom': 'Shortness of Breath', 'priority': 3},
    {'symptom': 'Muscle Pain', 'priority': 2},
    {'symptom': 'Chills', 'priority': 3},
    {'symptom': 'Loss of Taste', 'priority': 2},
    {'symptom': 'Loss of Smell', 'priority': 3},
    {'symptom': 'Nausea', 'priority': 1},
    {'symptom': 'Vomiting', 'priority': 2},
    {'symptom': 'Diarrhea', 'priority': 2},
    {'symptom': 'Abdominal Pain', 'priority': 2},
    {'symptom': 'Rash', 'priority': 2},
    {'symptom': 'Chest Pain', 'priority': 4},
    {'symptom': 'Dizziness', 'priority': 4},
    {'symptom': 'Confusion', 'priority': 3},
    {'symptom': 'Swollen Lymph Nodes', 'priority': 3},
    {'symptom': 'Joint Pain', 'priority': 2},
    {'symptom': 'Blurred Vision', 'priority': 4},
    {'symptom': 'Seizures', 'priority': 7},
    {'symptom': 'Difficulty Swallowing', 'priority': 7},
    {'symptom': 'Difficulty Breathing', 'priority': 3},
    {'symptom': 'Irritability', 'priority': 3},
    {'symptom': 'Memory Problems', 'priority': 4},
    {'symptom': 'Back Pain', 'priority': 2},
    {'symptom': 'Frequent Urination', 'priority': 3},
    {'symptom': 'Night Sweats', 'priority': 2},
    {'symptom': 'Unintended Weight Loss', 'priority': 5},
    {'symptom': 'Excessive Thirst', 'priority': 3},
    {'symptom': 'Excessive Hunger', 'priority': 3},
    {'symptom': 'Frequent Infections', 'priority': 5},
    {'symptom': 'Bruising Easily', 'priority': 4},
    {'symptom': 'Unexplained Bleeding', 'priority': 6},
    {'symptom': 'Swollen Joints', 'priority': 4},
    {'symptom': 'Chest Tightness', 'priority': 7},
    {'symptom': 'Pale Skin', 'priority': 5},
    {'symptom': 'Cold Hands and Feet', 'priority': 4},
    {'symptom': 'Hair Loss', 'priority': 3},
    {'symptom': 'Frequent Headaches', 'priority': 3},
    {'symptom': 'Nosebleeds', 'priority': 5},
    {'symptom': 'Frequent Sore Throat', 'priority': 3},
    {'symptom': 'Weakness', 'priority': 2},
    {'symptom': 'Yellowing of the Skin and Eyes (Jaundice)', 'priority': 7},
    {'symptom': 'Swollen Abdomen', 'priority': 8},
    {'symptom': 'Itchy Skin', 'priority': 4},
    {'symptom': 'Swollen Glands', 'priority': 4},
    {'symptom': 'Frequent Fever', 'priority': 4},
    {'symptom': 'Frequent Chills', 'priority': 5},
    {'symptom': 'Muscle Weakness', 'priority': 4},
    {'symptom': 'Dry Cough', 'priority': 1},
    {'symptom': 'Wheezing', 'priority': 7},
    {'symptom': 'Blood in Urine', 'priority': 8},
    {'symptom': 'Difficulty Urinating', 'priority': 7},
    {'symptom': 'Dribbling Urine', 'priority': 5},
    {'symptom': 'Testicular Pain', 'priority': 7},
    {'symptom': 'Erectile Dysfunction', 'priority': 5},
    {'symptom': 'Abnormal Vaginal Discharge', 'priority': 4},
    {'symptom': 'Painful Intercourse', 'priority': 3},
    {'symptom': 'Irregular Menstrual Periods', 'priority': 3},
    {'symptom': 'Unexplained Infertility', 'priority': 5},
    {'symptom': 'Lower Back Pain', 'priority': 4},
    {'symptom': 'Pelvic Pain', 'priority': 5},
    {'symptom': 'Frequent Hiccups', 'priority': 3},
    {'symptom': 'Difficulty Speaking', 'priority': 6},
    {'symptom': 'Slurred Speech', 'priority': 7},
    {'symptom': 'Tremors', 'priority': 5},
    {'symptom': 'Stiffness', 'priority': 5},
    {'symptom': 'Loss of Balance', 'priority': 4},
    {'symptom': 'Depression', 'priority': 6},
    {'symptom': 'Anxiety', 'priority': 6},
    {'symptom': 'Mood Swings', 'priority': 4},
    {'symptom': 'Sleep Problems', 'priority': 4},
    {'symptom': 'Feeling Hopeless', 'priority': 6},
    {'symptom': 'Loss of Interest', 'priority': 6},
    {'symptom': 'Suicidal Thoughts', 'priority': 7},
    {'symptom': 'Agitation', 'priority': 6},
    {'symptom': 'Poor Concentration', 'priority': 5},
    {'symptom': 'Panic Attacks', 'priority': 5},
    {'symptom': 'Rapid Heartbeat', 'priority': 7},
    {'symptom': 'Chest Discomfort', 'priority': 7},
    {'symptom': 'Numbness or Tingling', 'priority': 4},
    {'symptom': 'Swollen Ankles', 'priority': 5},
    {'symptom': 'Irregular Heartbeat', 'priority': 7},
    {'symptom': 'Coughing up Blood', 'priority': 8},
    {'symptom': 'Blood in Stool', 'priority': 8},
    {'symptom': 'Blood in Vomit', 'priority': 8},
    {'symptom': 'Unexplained Weight Gain', 'priority': 4},
    {'symptom': 'Heavy Menstrual Bleeding', 'priority': 4},
        {'symptom': 'Breast Changes', 'priority': 7},
    {'symptom': 'Nipple Discharge', 'priority': 7},
    {'symptom': 'Pelvic Pressure', 'priority': 8},
    {'symptom': 'Shortened Menstrual Periods', 'priority': 3},
    {'symptom': 'Hot Flashes', 'priority': 3},
    {'symptom': 'Night Sweats', 'priority': 5},
    {'symptom': 'Insomnia', 'priority': 2},
    {'symptom': 'Vaginal Dryness', 'priority': 2},
    {'symptom': 'Painful Urination', 'priority': 4},
    {'symptom': 'Painful Bowel Movements', 'priority': 3},
    {'symptom': 'Bloating', 'priority': 3},
    {'symptom': 'Abdominal Cramps', 'priority': 2},
    {'symptom': 'Feeling Full Quickly', 'priority': 2},
    {'symptom': 'Constant Hunger', 'priority': 2},
    {'symptom': 'Change in Bowel Habits', 'priority': 3},
    {'symptom': 'Swollen Legs', 'priority': 4},
    {'symptom': 'Swollen Hands', 'priority': 4},
    {'symptom': 'Frequent Bruising', 'priority': 5},
    {'symptom': 'Difficulty Walking', 'priority': 4},
    {'symptom': 'Balance Problems', 'priority': 4},
    {'symptom': 'Numbness in Extremities', 'priority': 4},
    {'symptom': 'Vision Changes', 'priority': 3},
    {'symptom': 'Hearing Loss', 'priority': 5},
    {'symptom': 'Ringing in Ears', 'priority': 5},
    {'symptom': 'Chest Pressure', 'priority': 7},
    {'symptom': 'Heartburn', 'priority': 3},
    {'symptom': 'Indigestion', 'priority': 2},
    {'symptom': 'Bloody Stool', 'priority': 8},
    {'symptom': 'Bloody Urine', 'priority': 8},
    {'symptom': 'Abnormal Bleeding', 'priority': 8},
    {'symptom': 'Bruising Under the Skin', 'priority': 8},
    {'symptom': 'Increased Urination at Night', 'priority': 5},
]


encoder = OneHotEncoder(sparse_output=False)
symptoms_encoded = encoder.fit_transform([[data['symptom']] for data in symptoms_data])
# Extract features and labels from the dataset
symptoms = [data['symptom'] for data in symptoms_data]
priorities = [data['priority'] for data in symptoms_data]

# Step 3: Assign Priorities and Sort Symptoms
model = build_decision_tree_model(symptoms_encoded, priorities)

# Example input with new patient symptoms
new_data = [
    "Fever", "Cough", "Headache", "Fatigue", "Sore Throat",
    "Shortness of Breath", "Muscle Pain", "Chills", "Loss of Taste",
    "Loss of Smell", "Nausea", "Vomiting", "Diarrhea", "Abdominal Pain",
    "Rash", "Chest Pain", "Dizziness", "Confusion", "Swollen Lymph Nodes",
    "Joint Pain", "Blurred Vision", "Seizures", "Difficulty Swallowing",
    "Difficulty Breathing", "Irritability", "Memory Problems", "Back Pain",
    "Frequent Urination", "Night Sweats", "Unintended Weight Loss",
    "Excessive Thirst", "Excessive Hunger", "Frequent Infections",
    "Bruising Easily", "Unexplained Bleeding", "Swollen Joints",
    "Chest Tightness", "Pale Skin", "Cold Hands and Feet", "Hair Loss",
    "Frequent Headaches", "Nosebleeds", "Frequent Sore Throat",
    "Weakness", "Yellowing of the Skin and Eyes (Jaundice)",
    "Swollen Abdomen", "Itchy Skin", "Swollen Glands", "Frequent Fever",
    "Frequent Chills", "Muscle Weakness", "Dry Cough", "Wheezing",
    "Blood in Urine", "Difficulty Urinating", "Dribbling Urine",
    "Testicular Pain", "Erectile Dysfunction", "Abnormal Vaginal Discharge",
    "Painful Intercourse", "Irregular Menstrual Periods",
    "Unexplained Infertility", "Lower Back Pain", "Pelvic Pain",
    "Frequent Hiccups", "Difficulty Speaking", "Slurred Speech",
    "Tremors", "Stiffness", "Loss of Balance", "Depression",
    "Anxiety"
]

# Reshape new_data into a 2D array with a single column
new_data_encoded = encoder.transform([[symptom] for symptom in new_data])

# Predict priorities for the new_data symptoms
predicted_priorities = model.predict(new_data_encoded)

# Combine symptoms with predicted priorities and sort based on priority and original order
symptoms_with_priorities = list(zip(new_data, predicted_priorities))
sorted_symptoms = sorted(symptoms_with_priorities, key=lambda x: (-x[1], symptoms.index(x[0])))

# Step 4: Generate Tokens
tokens = [f"Token_{i + 1}" for i in range(len(sorted_symptoms))]

print(sorted_symptoms)
print(tokens)
