# BarberoBot: A Hybrid RAG Semantic Digital Library Assistant

## Architecture

## Installation Instructions

## How to run the app locally
### Prerequisites 
* Python 3.10 or 3.11 installed on your system.
* A valid **Groq** API key to query the LLM.

### Download the project
You can either download the repository as a ZIP file and extract it, or clone it using your terminal:
```bash
git clone https://github.com/your-username/barberobot.git
```

### Set up a virtual environment
Create and activate a virtual environment to keep dependencies isolated:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file in the root directory of the project and add your Groq API key:
```env
GROQ_API_KEY="your_api_key_here"
```

### Run the application
Start the Chainlit server:
```bash
chainlit run app.py
```
The application will automatically open in your web browser at `http://localhost:8000`.

## Copyright Disclaimer 
