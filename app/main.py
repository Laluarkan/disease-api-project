from fastapi import FastAPI
from app.schema import GejalaInput, PenyakitInput
import pickle
import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
import os

BASE_DIR = os.path.dirname(__file__)  # Lokasi file main.py
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
with open(LABEL_ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_features.txt"), "r") as f:
    trained_features = [line.strip() for line in f]

df = pd.read_csv(os.path.join(BASE_DIR, "Training (1).csv"))

# === Mapping Gejala ke Bahasa Indonesia ===
# (isi mapping_indonesia lengkap sesuai sebelumnya)
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

mapping_penyakit_indonesia = {
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

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
def baca_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/prediksi-awal")
def prediksi_awal(data: GejalaInput):
    input_dict = dict.fromkeys(trained_features, 0)
    for key, val in data.jawaban.items():
        if key in input_dict:
            input_dict[key] = val

    input_array = np.array([[input_dict[feat] for feat in trained_features]])
    proba = model.predict_proba(input_array)[0]
    top10_idx = np.argsort(proba)[-10:][::-1]
    top10_penyakit = encoder.inverse_transform(top10_idx)
    top1_penyakit = top10_penyakit[0]

    top10_penyakit_ind = [mapping_penyakit_indonesia.get(p, p) for p in top10_penyakit]
    top1_penyakit_ind = mapping_penyakit_indonesia.get(top1_penyakit, top1_penyakit)

    return {
        "prediksi_teratas": top10_penyakit_ind,
        "penyakit_top1": top1_penyakit_ind
    }


@app.post("/pertanyaan-lanjutan")
def pertanyaan_lanjutan(data: PenyakitInput):
    # Konversi penyakit dari Indonesia ke Inggris
    mapping_indonesia_ke_inggris = {v: k for k, v in mapping_penyakit_indonesia.items()}
    penyakit_ind = data.nama_penyakit
    penyakit_inggris = mapping_indonesia_ke_inggris.get(penyakit_ind, penyakit_ind)

    if penyakit_inggris not in df["prognosis"].unique():
        return {"error": f"Penyakit '{penyakit_ind}' tidak ditemukan di data."}

    # Ambil gejala yang paling sering muncul pada penyakit tersebut
    gejala_sum = df[df["prognosis"] == penyakit_inggris].drop(columns=["prognosis"]).sum()
    gejala_teratas = gejala_sum[gejala_sum > 0].sort_values(ascending=False).index.tolist()[:10]

    pertanyaan = []
    for g in gejala_teratas:
        label = mapping_indonesia.get(g, g.replace("_", " "))
        pertanyaan.append(f"Apakah Anda mengalami {label}? (y/n)")

    return {
        "penyakit": penyakit_ind,  # Tetap tampilkan nama Indonesia di output
        "pertanyaan_tambahan": pertanyaan
    }


@app.post("/prediksi-akhir")
def prediksi_akhir(data: GejalaInput):
    input_dict = dict.fromkeys(trained_features, 0)
    for key, val in data.jawaban.items():
        if key in input_dict:
            input_dict[key] = val

    input_array = np.array([[input_dict[feat] for feat in trained_features]])
    pred = model.predict(input_array)[0]
    final_penyakit = encoder.inverse_transform([pred])[0]

    final_penyakit_ind = mapping_penyakit_indonesia.get(final_penyakit, final_penyakit)

    return {
        "prediksi_akhir": final_penyakit_ind
    }
