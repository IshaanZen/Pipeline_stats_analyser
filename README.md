# 🚀 Databricks Pipeline Stats Analyser

A smart, business-friendly dashboard that converts complex Databricks pipeline performance data into easy-to-understand insights for managers, project stakeholders, and non-technical decision makers.

This project supports **two types of inputs**:

* 📊 **Structured Data (CSV / JSON)** – Traditional pipeline metrics files
* 📸 **Pipeline Screenshots (Images)** – UI screenshots from Databricks Cluster / Spark UI

Behind the scenes, the app uses AI to:

* Extract metrics (even from images)
* Summarise pipeline health
* Provide plain-English explanations for business use

---

## ✨ Key Features

### ✅ Structured Data Mode

* Upload CSV or JSON with pipeline metrics
* Interactive filters by pipeline
* Visual Charts

  * CPU Utilisation per run
  * Memory Usage per run
* Aggregate statistics

  * Average runtime
  * Average CPU & Memory usage
* AI-generated explanations of pipeline performance

### ✅ Image Mode (AI Vision-powered)

* Upload two Databricks screenshots:

  1. Cluster / Driver Stats
  2. Only Executor Nodes Stats
* AI extracts approximate metrics
* Converts visual graphs into structured JSON
* Generates a business-friendly performance summary

---

## 🧠 Who is this for?

* Project Managers
* Business Stakeholders
* Analysts
* Non-technical Leadership
* Data Engineering Teams presenting reports

---

## 🛠️ Tech Stack

* **Streamlit** – Web UI framework
* **Python** – Core logic
* **OpenRouter API** – AI layer (Vision + Text)
* **Pandas** – Data processing
* **dotenv** – Secure configuration handling

---

## 📁 Project Structure

```
pipeline-stats-analyser/
│
├── app.py
├── data/
│   └── sample_data.json
├── .env                # Not committed to Git
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔐 Environment Setup

Create a `.env` file in the root folder:

```
OPENROUTER_API_KEY=your_api_key_here
```

This file is ignored using `.gitignore` to keep your key safe.

---

## ▶️ How to Run Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

If you don’t have a requirements file yet:

```
pip install streamlit pandas python-dotenv openai
```

### 2. Start the app

```
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 📊 Example Use Cases

* Weekly pipeline performance review meeting
* Client presentation of Databricks job efficiency
* Quick troubleshooting overview for management
* Visual-to-report automation

---

## 🔮 Future Enhancements

* Export report as PDF
* Slack / Email alerts
* Historical trend comparison
* Auto anomaly detection
* Integration with Databricks API

---

## 👨‍💻 Author

Built with 💙 by a Databricks Data Engineering Intern
for real-world business enablement and performance clarity.

---

## ⭐ Star the repo if you find it useful!

This project aims to bridge the gap between deep technical systems and business understanding.
