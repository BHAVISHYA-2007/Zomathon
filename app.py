import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load dataset with error handling
try:
    df = pd.read_csv("orders.csv")
    
    # Convert to transaction list
    transactions = df['Items'].apply(lambda x: [item.strip() for item in str(x).split(",")]).tolist()
    
    # One-hot encoding
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    
    # Apply Apriori
    frequent_itemsets = apriori(df_encoded, min_support=0.15, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    else:
        rules = pd.DataFrame()
except Exception as e:
    print(f"Warning: Could not load AI model - {e}")
    rules = pd.DataFrame()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        user_items = data.get("items", []) if data else []
        
        recommendations = []
        
        if len(rules) > 0 and user_items:
            for _, row in rules.iterrows():
                if set(user_items).issubset(row['antecedents']):
                    recommendations.extend(list(row['consequents']))
        
        # Remove duplicates and limit results
        unique_recs = list(set(recommendations))[:10]
        
        return jsonify({
            "success": True,
            "recommendations": unique_recs,
            "count": len(unique_recs)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "recommendations": [],
            "error": str(e)
        }), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ai_model_loaded": len(rules) > 0})

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        data = request.get_json()
        # Preferences are stored client-side in localStorage
        # This endpoint acknowledges the update and could trigger model retraining if needed
        preferences = {
            "spice": data.get("spice", 78),
            "sweet": data.get("sweet", 55),
            "healthy": data.get("healthy", 40),
            "veg": data.get("veg", 45),
            "budget": data.get("budget", 60)
        }
        
        # Log preferences for future use
        print(f"User preferences updated: {preferences}")
        
        return jsonify({
            "success": True,
            "message": "AI preferences updated successfully",
            "preferences": preferences
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)