# categories.py — Fixed & Working
import re
import requests
import pandas as pd
from typing import List, Optional, Tuple
import os 
from dotenv import load_dotenv


load_dotenv("pass.env")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")



# ---------- Rules ----------
Rules = [
    # RENT - Housing and accommodation
    (r"\b(RENT|RENTAL|HOUSE RENT|MAINTENANCE|SECURITY DEPOSIT|SOCIETY FEES|BROKERAGE|LEASE|TENANT|PG|HOSTEL|RENT AGREEMENT|HOUSEHOLD|ACCOMMODATION)\b", "RENT"),
    
    # UTILITIES - Essential services
    (r"\b(ELECTRICITY|POWER|WATER BILL|WATER|INTERNET|BROADBAND|WIFI|DTH|MOBILE BILL|RECHARGE|AIRTEL|BSNL|JIO|TATA PLAY|GAS|LPG|COOK|CLEANING|UTILITIES|BILL|UTILITY)\b", "UTILITIES"),
    
    # TRANSPORT - Travel and transportation
    (r"\b(TRANSPORT|UBER|RAPIDO|OLA|AUTO|CAB|TAXI|IRCTC|FUEL|PETROL|PETROLEUM|TRAIN|FLIGHT|BUS|METRO|PARKING|TOLL|TRAVEL|COMMUTE)\b", "TRANSPORT"),
    
    # FOOD & BEVERAGES - Dining and drinks
    (r"\b(ZOMATO|SWIGGY|STARBUCKS|KFC|DOMINOS|EATCLUB|MCDONALDS|RESTAURANT|FOOD|LIQUOR|BEER|WHISKY|WINE|RUM|VODKA|GIN|TEQUILA|THUMS UP|BEVERAGES|DINING|EAT|DRINK|CAFE|COFFEE|TEA)\b", "FOOD & BEVERAGES"),
    
    # GROCERIES - Shopping for essentials
    (r"\b(GROCERY|GROCERIES|DMART|BIG BAZAAR|FLIPKART|ZEPTO|BLINKIT|RELIANCE FRESH|BIG BASKET|INSTAMART|LISCIOUS|SPAR|MORE|SUPERMARKET|NATURE'S BASKET|GROCERY|ESSENTIALS)\b", "GROCERIES"),
    
    # BILLS & SUBSCRIPTIONS - Recurring payments
    (r"\b(SPOTIFY|NETFLIX|AMAZON PRIME|ICLOUD|SUBSCRIPTION|GOOGLE|YOUTUBE PREMIUM|GEMINI|BILLS|SUBSCRIPTION|RECURRING|MONTHLY|ANNUAL)\b", "BILLS & SUBSCRIPTIONS"),
    
    # HEALTHCARE - Medical and health
    (r"\b(INSURANCE|HEALTH|FITNESS|MEDICAL|MEDPLUS|APOLLO PHARMACY|HOSPITAL|LAB TEST|PHARMACY|HEALTH INSURANCE|TATA 1MG|DOCTOR|CLINIC|MEDICINE|HEALTH CHECKUP|HEALTHCARE|MEDICAL|PHARMACY)\b", "HEALTHCARE"),
    
    # SHOPPING - Retail purchases
    (r"\b(SHOPPING|CROMA|RELIANCE DIGITAL|DECATHLON|ELECTRONICS|AMAZON|MYNTRA|DISTRICT|AJIO|H&M|SHOP|PURCHASE|BUY|RETAIL|STORE|MALL)\b", "SHOPPING"),
    
    # TRAVEL - Tourism and leisure
    (r"\b(TRAVEL|MAKE MY TRIP|HOTEL|AIRBNB|OYO|BOOKING|TRAVEL|VACATION|HOLIDAY|TOURISM|LEISURE|TRIP)\b", "TRAVEL"),
    
    # INCOME/REFUND - Money coming in
    (r"\b(INCOME|SALARY|WAGE|BONUS|REFUND|CREDIT|DEPOSIT|TRANSFER IN|CASHBACK|REWARD|CASHBACK|INCOME|REFUND)\b", "INCOME/REFUND"),
    
    # FEES/CHARGES - Service charges
    (r"\b(FEE|CHARGE|ATM|PENALTY|FINE|SERVICE CHARGE|PROCESSING FEE|TRANSACTION FEE|BANK CHARGE|MAINTENANCE FEE)\b", "FEES/CHARGES"),
    
    # MISCELLANEOUS - Everything else (this should be the last rule)
    (r".*", "MISCELLANEOUS")
]


class Classifier:
    def __init__(
        self,
        categories: List[str],
        api_key: Optional[str] = None,
        model_name: str = "openrouter/auto",
        rule_score: float = 0.9,
        llm_threshold: float = 0.8,
        unknown_label: str = "Uncategorized",
        enable_llm: bool = True,
    ):
        self.categories = categories
        self.api_key = api_key or OPENROUTER_KEY
        self.model = model_name
        self.rule_score = rule_score
        self.llm_threshold = llm_threshold
        self.unknown_label = unknown_label
        # enable LLM if either explicit api_key provided or loaded from env
        self.enable_llm = enable_llm and bool(self.api_key)

    # ---------- Rule-based Matching ----------
    def match(self, text: str) -> Tuple[Optional[str], float]:
        clean = (text or "").upper()
        for pattern, label in Rules:
            if re.search(pattern, clean):
                if label in self.categories:
                    return label, self.rule_score
        return None, 0.0

    # ---------- LLM-based Fallback ----------
    def llm(self, description: str) -> Tuple[Optional[str], float]:
        if not self.enable_llm or not self.api_key:
            return None, 0.0

        system_prompt = (
            "You are a precise financial transaction classifier. "
            "Assign each transaction to exactly one category from the list below.\n"
            "Categories: " + ", ".join(self.categories) + "\n"
            "Guidelines:\n"
            "- Choose the most appropriate single category.\n"
            "- If unclear, return 'Uncategorized'.\n"
            "- Respond only with the category name (case-sensitive)."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transaction description: {description.strip()}"},
            ],
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Recommended by OpenRouter for browser or web-app usage
            "HTTP-Referer": "http://localhost",
            "X-Title": "Expense Categorizer",
        }

        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json=payload, headers=headers, timeout=20)
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content in self.categories:
                return content, 0.8
            return self.unknown_label, 0.5
        except Exception as e:
            print("LLM classification failed:", e)
            return None, 0.0

    # ---------- Apply to DataFrame ----------
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df["Description"].apply(lambda x: self.match(x))
        df[["Category", "Confidence"]] = pd.DataFrame(results.tolist(), index=df.index)
        return df