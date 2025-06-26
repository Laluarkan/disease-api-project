import pickle
import numpy as np
import pandas as pd

# === Muat model dan encoder ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# === Muat urutan fitur dari model_features.txt ===
with open("model_features.txt", "r") as f:
    trained_features = [line.strip() for line in f]

# === Inisialisasi semua fitur bernilai 0 ===
input_dict = dict.fromkeys(trained_features, 0)

# === Gejala awal yang akan ditanyakan ===
gejala_awal = [
    "fever", "cough", "headache", "fatigue", "muscle_pain",
    "nausea", "vomiting", "dizziness", "diarrhoea", "itching"
]

# Mapping nama fitur ke Bahasa Indonesia
mapping_indonesia = {
"fever": "demam",
"cough": "batuk",
"headache": "sakit kepala",
"fatigue": "kelelahan",
"muscle_pain": "nyeri otot",
"nausea": "mual",
"vomiting": "muntah",
"dizziness": "pusing",
"diarrhoea": "diare",
"itching": "gatal",
"abdominal_pain": "sakit perut",
"abnormal_menstruation": "menstruasi tidak normal",
"acidity": "asam lambung",
"acute_liver_failure": "gagal hati akut",
"altered_sensorium": "sensorium berubah",
"anxiety": "kecemasan",
"back_pain": "sakit punggung",
"belly_pain": "nyeri perut",
"blackheads": "komedo",
"bladder_discomfort": "ketidaknyamanan kandung kemih",
"blister": "lepuh",
"blood_in_sputum": "darah dalam dahak",
"bloody_stool": "tinja berdarah",
"blurred_and_distorted_vision": "penglihatan kabur dan terdistorsi",
"breathlessness": "sesak napas",
"brittle_nails": "kuku rapuh",
"bruising": "memar",
"burning_micturition": "buang air kecil terasa panas",
"chest_pain": "nyeri dada",
"chills": "menggigil",
"cold_hands_and_feet": "tangan dan kaki dingin",
"coma": "koma",
"congestion": "hidung tersumbat",
"constipation": "sembelit",
"continuous_feel_of_urine": "terus-menerus merasa ingin buang air kecil",
"continuous_sneezing": "bersin terus-menerus",
"cough": "batuk",
"cramps": "kram",
"dark_urine": "urin gelap",
"dehydration": "dehidrasi",
"depression": "depresi",
"diarrhoea": "diare",
"dyschromic_patches": "bercak dischromic",
"distention_of_abdomen": "perut kembung",
"dizziness": "pusing",
"drying_and_tingling_lips": "bibir kering dan kesemutan",
"enlarged_thyroid": "tiroid membesar",
"excessive_hunger": "rasa lapar berlebihan",
"extra_marital_contacts": "kontak di luar nikah",
"family_history": "riwayat keluarga",
"fast_heart_rate": "detak jantung cepat",
"fatigue": "kelelahan",
"fluid_overload": "kelebihan cairan",
"fluid_overload.1": "kelebihan cairan",
"foul_smell_of urine": "bau urin menyengat",
"headache": "sakit kepala",
"high_fever": "demam tinggi",
"hip_joint_pain": "nyeri sendi pinggul",
"history_of_alcohol_consumption": "riwayat konsumsi alkohol",
"increased_appetite": "nafsu makan meningkat",
"indigestion": "gangguan pencernaan",
"inflammatory_nails": "kuku meradang",
"internal_itching": "gatal internal",
"irregular_sugar_level": "kadar gula tidak teratur",
"irritability": "iritabilitas",
"irritation_in_anus": "iritasi di anus",
"itching": "gatal",
"joint_pain": "nyeri sendi",
"knee_pain": "nyeri lutut",
"lack_of_concentration": "kurang konsentrasi",
"lethargy": "lesu",
"loss_of_appetite": "kehilangan nafsu makan",
"loss_of_balance": "kehilangan keseimbangan",
"loss_of_smell": "kehilangan penciuman",
"loss_of_taste": "kehilangan rasa",
"malaise": "malaise",
"mild_fever": "demam ringan",
"mood_swings": "perubahan suasana hati",
"movement_stiffness": "kekakuan gerakan",
"mucoid_sputum": "dahak mukoid",
"muscle_pain": "nyeri otot",
"muscle_wasting": "atrofi otot",
"muscle_weakness": "kelemahan otot",
"nausea": "mual",
"neck_pain": "nyeri leher",
"nodal_skin_eruptions": "erupsi kulit nodal",
"obesity": "obesitas",
"pain_behind_the_eyes": "nyeri di belakang mata",
"pain_during_bowel_movements": "nyeri saat buang air besar",
"pain_in_anal_region": "nyeri di daerah anus",
"painful_walking": "sulit berjalan",
"palpitations": "palpitasi",
"passage_of_gases": "keluarnya gas",
"patches_in_throat": "bercak di tenggorokan",
"phlegm": "dahak",
"polyuria": "poliuria",
"prominent_veins_on_calf": "vena menonjol di betis",
"puffy_face_and_eyes": "wajah dan mata bengkak",
"pus_filled_pimples": "jerawat berisi nanah",
"receiving_blood_transfusion": "menerima transfusi darah",
"receiving_unsterile_injections": "menerima suntikan tidak steril",
"red_sore_around_nose": "luka merah di sekitar hidung",
"red_spots_over_body": "bintik merah di seluruh tubuh",
"redness_of_eyes": "mata merah",
"restlessness": "gelisah",
"runny_nose": "hidung meler",
"rusty_sputum": "dahak berkarat",
"scurrying": "bergerak cepat",
"shivering": "menggigil",
"silver_like_dusting": "debu seperti perak",
"sinus_pressure": "tekanan sinus",
"skin_peeling": "kulit mengelupas",
"skin_rash": "ruam kulit",
"slurred_speech": "bicara cadel",
"small_dents_in_nails": "lekukan kecil di kuku",
"spinning_movements": "gerakan berputar",
"spotting_urination": "bercak urin",
"stiff_neck": "leher kaku",
"stomach_bleeding": "pendarahan perut",
"stomach_pain": "sakit perut",
"sunken_eyes": "mata cekung",
"sweating": "berkeringat",
"swelled_lymph_nodes": "kelenjar getah bening bengkak",
"swelling_joints": "sendi bengkak",
"swelling_of_stomach": "pembengkakan perut",
"swollen_blood_vessels": "pembuluh darah bengkak",
"swollen_extremities": "ekstremitas bengkak",
"swollen_legs": "kaki bengkak",
"throat_irritation": "iritasi tenggorokan",
"tiredness": "kelelahan",
"toxic_look_(typhus)": "tampilan toksik (tifus)",
"ulcers_on_tongue": "bisul di lidah",
"unsteadiness": "ketidakstabilan",
"visual_disturbances": "gangguan penglihatan",
"vomiting": "muntah",
"watering_from_eyes": "mata berair",
"weakness_in_limbs": "kelemahan pada anggota badan",
"weakness_of_one_body_side": "kelemahan pada satu sisi tubuh",
"weight_gain": "penambahan berat badan",
"weight_loss": "penurunan berat badan",
"yellow_crust_ooze": "nanah kuning berkerak",
"yellow_urine": "urin kuning",
"yellowing_of_eyes": "mata menguning",
"yellowish_skin": "kulit kekuningan"
}

mapping_penyakit = {
    "AIDS": "AIDS",
"Acne": "Jerawat",
"Alcoholic hepatitis": "Hepatitis alkoholik",
"Allergy": "Alergi",
"Arthritis": "Artritis",
"Bronchial Asthma": "Asma Bronkial",
"Cervical spondylosis": "Spondilosis servikal",
"Chicken pox": "Cacar air",
"Chronic cholestasis": "Kolestasis kronis",
"Common Cold": "Flu biasa",
"Covid": "Covid",
"Dengue": "Demam berdarah",
"Diabetes": "Diabetes",
"Dimorphic hemorrhoids (piles)": "Wasir dimorfik",
"Drug Reaction": "Reaksi obat",
"Fungal infection": "Infeksi jamur",
"GERD": "GERD",
"Gastroenteritis": "Gastroenteritis",
"Heart attack": "Serangan jantung",
"Hepatitis A": "Hepatitis A",
"Hepatitis B": "Hepatitis B",
"Hepatitis C": "Hepatitis C",
"Hepatitis D": "Hepatitis D",
"Hepatitis E": "Hepatitis E",
"Hypertension": "Hipertensi",
"Hyperthyroidism": "Hipertiroidisme",
"Hypoglycemia": "Hipoglikemia",
"Hypothyroidism": "Hipotiroidisme",
"Impetigo": "Impetigo",
"Jaundice": "Penyakit kuning",
"Malaria": "Malaria",
"Migraine": "Migrain",
"Osteoarthritis": "Osteoartritis",
"Paralysis (brain hemorrhage)": "Kelumpuhan (pendarahan otak)",
"Paroxysmal Positional Vertigo": "Vertigo Positional Paroksismal",
"Peptic ulcer disease": "Penyakit tukak lambung",
"Pneumonia": "Pneumonia",
"Psoriasis": "Psoriasis",
"Tuberculosis": "Tuberkulosis",
"Typhoid": "Tipes",
"Urinary tract infection": "Infeksi saluran kemih",
"Varicose veins": "Varises"
}


# === Input jawaban awal ===
print("❓ Jawab pertanyaan berikut (y/n):")
for gejala in gejala_awal:
    label_indo = mapping_indonesia.get(gejala, gejala.replace('_', ' '))
    jawab = input(f"Apakah Anda mengalami {label_indo}? (y/n): ").lower()
    if jawab == "y":
        input_dict[gejala] = 1

if sum(input_dict.values()) == 0:
    print("\n⚠️ Anda tidak memilih gejala apa pun. Mohon isi setidaknya satu gejala untuk mendapatkan prediksi.")
    exit()

# === Prediksi awal ===
input_array = np.array([[input_dict[feat] for feat in trained_features]])
proba = model.predict_proba(input_array)[0]
top10_idx = np.argsort(proba)[-10:][::-1]
top10_penyakit = le.inverse_transform(top10_idx)

print("\n🧪 Prediksi awal (kemungkinan):")
for penyakit in top10_penyakit:
    print(f"- {mapping_penyakit.get(penyakit, penyakit)}")

# === Gejala tambahan berdasarkan penyakit pertama ===
penyakit_top1 = top10_penyakit[0]
df = pd.read_csv("Training (1).csv")
gejala_penting = df[df["prognosis"] == penyakit_top1].drop(columns=["prognosis"]).sum()
gejala_penting = gejala_penting[gejala_penting > 0].sort_values(ascending=False).index.tolist()

# Filter gejala tambahan yang belum ditanya
gejala_tambahan = [g for g in gejala_penting if g not in gejala_awal][:10]

print("\n📋 Pertanyaan tambahan berdasarkan prediksi awal:")
for gejala in gejala_tambahan:
    label_indo = mapping_indonesia.get(gejala, gejala.replace('_', ' '))
    jawab = input(f"Apakah Anda mengalami {label_indo}? (y/n): ").lower()
    if jawab == "y":
        input_dict[gejala] = 1

# === Prediksi akhir ===
final_input = np.array([[input_dict[feat] for feat in trained_features]])
final_pred = model.predict(final_input)[0]
final_penyakit = le.inverse_transform([final_pred])[0]

penyakit_indo = mapping_penyakit.get(final_penyakit, final_penyakit)
print("\n✅ Berdasarkan jawaban Anda, kemungkinan penyakit Anda adalah:", penyakit_indo)

