import requests

url = "http://0.0.0.0:4444/predict_cpu_tf_rf"
#url = "https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf_rf"

observation = {
    "cpu_cores": 6,
    "cpu_threads": 12,
    "cpu_frq": 3.6,
    "cpu_clock": 0.1,
    "cpu_multiplier": 35,
    "cpu_tdp": 65,
    "cpu_prcss": 14,
    "cpu_die": 160,
    "cpu_has_oc": "N"
}

response = requests.post(url, json=observation)
result = response.json()

print(f"Predicted turbo frequency: ~{result.get('turbo_frequency')} GHz")