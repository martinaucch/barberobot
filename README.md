# BarberoBot: A Hybrid RAG Semantic Digital Library Assistant

## Features
* **Transparent Sourcing**: Whenever BarberoBot answers, it displays the exact transcription segments used to generate the response in a dedicated side panel.
* **Direct Archive Integration** Responses automatically include direct links to the Barberotheca website, allowing you to seamlessy open and listen or read the original lesson. 

## Architecture

## Deployment (Live Demo)

To allow immediate testing without local installation, BarberoBot has been deployed. You can interact with the live assistant here:

**[Access BarberoBot Live](https://barberobot.onrender.com/)**

>[!NOTE]
>* **Hosting Platform:** The application is hosted on [Render](https://render.com) using their free tier.
>* **Cold Starts:** Because it runs on a free instance, the server goes to sleep after a period of inactivity. If the bot hasn't been accessed recently, it may take 1 to 2 minutes to
"wake up" and load upon your first visit. 

## How to run the app locally
### Prerequisites 
>[!IMPORTANT]
> Before starting, ensure your system meets the following requirements and that you have the necessary API keys:

* **Python 3.10 or 3.11**: Newer versions (e.g., Python 3.14) are not currently supported due to known incompatibilities with the asynchronous `anyio` library used by the interface.
* **Git LFS (Large File Storage)**: Required to download the pre-computed ChromaDB vector database. We chose to provide the database via LFS to save users from having to re-vectorize the entire archive.
  * *On Mac*: Install via terminal by typing `brew install git-lfs`
  * *On Windows/Linux*: Download and install from [git-lfs.github.com](https://git-lfs.github.com/)
* **API Keys**: The hybrid cloud system requires you to create two free API keys:
  * **Groq API Key**: It has been decided to keep this step to leverage Groq's lightning-fast infrastructure and the *Qwen* LLM (`qwen/qwen3.8-27b`), which the tests guaranteed significantly better response quality.
  * **Cohere API Key**: Required as the engine for processing multilingual embeddings and for the crucial cross-encoder re-ranking step.

### Clone the project
Open your terminal and clone the repository:
```bash
git clone https://github.com/martinaucch/barberobot.git
```
>[!NOTE]
>The submodule initialization may take a moment as it downloads the complete original text transcripts alongside the metadata. While these raw transcripts are not strictly required to >run the app right now (since the vector database is already pre-computed and provided via Git LFS), pulling them is a deliberate architectural choice. It ensures the environment >remains fully scalable, and ready for any future local re-ingestion or pipeline updates.

### Data synchronization
Download the large vector database files and initialize the Barberotheca submodule to gain access to the local RDF graph files:
```bash
git lfs pull
git submodule update --init --recursive
```

### Set up virtual environment
Create and activate a virtual environment to keep dependencies isolated:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install dependencies
Install the project dependencies:
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file in the root directory of the project and add your Groq and Cohere API key:
```env
GROQ_API_KEY="your_groq_api_key_here"
COHERE_API_KEY="your_cohere_api_key_here
```

### Run the application
Start the Chainlit server:
```bash
chainlit run app.py
```
The application will automatically open in your web browser at `http://localhost:8000`.

## Copyright Disclaimer 
