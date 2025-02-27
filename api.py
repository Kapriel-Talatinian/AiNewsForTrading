from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import requests
import openai
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_news(target_datetime):
    """Récupère les actualités avec gestion d'erreur améliorée"""
    try:
        app.logger.debug("Recherche d'actualités pour %s", target_datetime)
        
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "NASDAQ OR US100",
                "from": (target_datetime - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                "to": target_datetime.strftime("%Y-%m-%dT%H:%M:%S"),
                "apiKey": NEWSAPI_KEY,
                "pageSize": 5,
                "language": "en",
                "sortBy": "publishedAt"
            },
            timeout=8
        )
        
        if response.status_code != 200:
            app.logger.error("Erreur NewsAPI: %s", response.text)
            return []
            
        articles = response.json().get('articles', [])[:2]
        app.logger.debug("%d articles trouvés", len(articles))
        return articles
        
    except Exception as e:
        app.logger.error("Erreur fetch_news: %s", str(e))
        return []

def analyze_news(articles):
    """Analyse avec GPT-4 optimisé"""
    try:
        if not articles:
            return {"recommendation": "HOLD", "confidence": 0.0, "reason": "Aucune actualité"}
            
        messages = [{
            "role": "system",
            "content": """Analyse les actualités financières. Réponds en JSON avec:
            - recommendation: BUY/SELL/HOLD
            - confidence: 0.0-1.0
            - reason: explication concise en 50 mots max"""
        }]
        
        for idx, article in enumerate(articles[:3]):
            content = f"{article.get('title', '')}\n{article.get('description', '')}"
            messages.append({
                "role": "user",
                "content": f"Actualité {idx+1}: {content[:450]}"  # max toekn
            })
        
        app.logger.debug("Envoi à GPT-4: %s", str(messages))
        
        #GPT CONFIG
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=75
        )
        
        raw_response = response.choices[0].message.content
        app.logger.debug("Réponse GPT-4 brute: %s", raw_response)
        
        analysis = json.loads(raw_response)
        analysis["confidence"] = max(0.0, min(1.0, float(analysis.get("confidence", 0.0))))
        
        return analysis
        
    except json.JSONDecodeError:
        app.logger.error("Réponse GPT-4 invalide: %s", raw_response)
        return {"recommendation": "HOLD", "confidence": 0.0, "reason": "Erreur d'analyse"}
    except Exception as e:
        app.logger.error("Erreur analyze_news: %s", str(e))
        return {"recommendation": "HOLD", "confidence": 0.0, "reason": str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint principal avec tracing complet"""
    start_time = datetime.utcnow()
    
    try:
        app.logger.info("Nouvelle requête reçue")
        data = request.get_json()
        
        if not data or 'datetime' not in data:
            app.logger.warning("Requête invalide: %s", data)
            return jsonify({"error": "Champ datetime requis"}), 400
            
        try:
            target_datetime = datetime.fromisoformat(data['datetime'].replace("Z", ""))
        except ValueError as e:
            app.logger.error("Format de date invalide: %s", data['datetime'])
            return jsonify({"error": f"Format datetime invalide: {str(e)}"}), 400
        
        articles = fetch_news(target_datetime)
        
        analysis = analyze_news(articles)
        
        response_data = {
            "recommendation": analysis.get("recommendation", "HOLD").upper(),
            "confidence": analysis["confidence"],
            "reason": analysis.get("reason", "Non spécifié"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_time": (datetime.utcnow() - start_time).total_seconds()
        }
        
        app.logger.info("Réponse générée: %s", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.exception("Erreur critique: %s", str(e))
        return jsonify({
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reason": "Erreur serveur",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)