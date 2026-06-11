"""
Smart Cotton Assistant Chatbot.
Supports rule-based knowledge base and Gemini API integration.
Includes multilingual responses (English, Hindi, Marathi).
"""

import os
from typing import Optional, Tuple

import config

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Rule-based knowledge base (English)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_EN = {
    "bacterial blight": (
        "Bacterial Blight is a serious cotton disease caused by Xanthomonas citri "
        "subsp. malvacearum. It appears as angular water-soaked lesions on leaves "
        "that turn brown with yellow halos. Prevention includes using disease-free seeds, "
        "crop rotation, and copper-based sprays. Treatment involves Streptomycin sulphate "
        "or Copper oxychloride applications at 10-15 day intervals."
    ),
    "curl virus": (
        "Cotton Leaf Curl Virus (CLCuV) is transmitted by whiteflies (Bemisia tabaci). "
        "Symptoms include upward leaf curling, thickening, and stunted growth. "
        "There is no direct cure for viral infections. Control whitefly vectors using "
        "Imidacloprid, install yellow sticky traps, and grow resistant varieties like PKV 081."
    ),
    "fusarium wilt": (
        "Fusarium Wilt is caused by Fusarium oxysporum, a soil-borne fungus. "
        "Symptoms include yellowing of lower leaves, wilting during hot hours, and "
        "brown vascular discoloration. Prevention: use resistant varieties, maintain "
        "soil pH 6.0-7.0, practice 4-5 year crop rotation, and treat seeds with Trichoderma."
    ),
    "healthy": (
        "A healthy cotton plant shows uniform green leaves without spots, lesions, or "
        "discoloration. Maintain regular monitoring, balanced NPK fertilization, proper "
        "spacing, and crop rotation to keep plants healthy."
    ),
    "irrigation": (
        "Cotton requires 500-700 mm water during its growing season. Critical irrigation "
        "stages: sowing, flowering, and boll formation. Use drip irrigation for water "
        "efficiency. Avoid waterlogging which promotes Fusarium wilt. In India, irrigate "
        "every 10-15 days depending on soil type and rainfall."
    ),
    "fertilizer": (
        "Recommended NPK for cotton: 100-150 kg N, 50-60 kg P2O5, and 50-60 kg K2O per hectare. "
        "Apply nitrogen in split doses: 1/3 at sowing, 1/3 at squaring, 1/3 at flowering. "
        "Use FYM (10 t/ha) before sowing. Apply micronutrients (Zn, B) if deficiency observed. "
        "Soil testing is essential for precise fertilizer recommendations."
    ),
    "pest": (
        "Major cotton pests: bollworm (Helicoverpa), whitefly, aphids, jassids, and mealybugs. "
        "Integrated Pest Management (IPM) approach: monitor with pheromone traps, use "
        "bio-pesticides (NPV, Bt), apply chemical pesticides only above economic threshold. "
        "Neem oil 5% spray is effective against sucking pests. Maintain field sanitation."
    ),
    "farming practices": (
        "Best cotton farming practices: 1) Select certified seeds of recommended varieties. "
        "2) Prepare fine seedbed with proper drainage. 3) Sow at 75x30 cm spacing. "
        "4) Apply balanced fertilizers based on soil test. 5) Regular pest and disease scouting. "
        "6) Timely irrigation especially during flowering. 7) Harvest when 80% bolls open. "
        "8) Practice crop rotation with cereals or legumes."
    ),
    "crop maintenance": (
        "Crop maintenance tips: Thinning at 15-20 days after sowing. First weeding at 20-25 DAS, "
        "second at 45-50 DAS. Pinching/pruning for better branching. Monitor for pests weekly. "
        "Remove diseased plants immediately. Apply growth regulators if needed. "
        "Keep field borders clean of weeds that harbor pests and diseases."
    ),
    "sowing": (
        "Optimal sowing time for cotton in India: June-July (kharif season). "
        "Seed rate: 12-15 kg/ha for hybrids, 15-20 kg/ha for varieties. "
        "Sowing depth: 3-5 cm. Soil temperature should be above 18°C. "
        "Treat seeds with fungicide (Carbendazim 2g/kg) and bio-agent (Trichoderma)."
    ),
    "harvest": (
        "Harvest cotton when 80-90% bolls are fully open. Picking should be done in 3-4 pickings "
        "at 15-20 day intervals. Harvest early morning when moisture is higher (better fiber quality). "
        "Dry picked cotton in shade before storage. Store in clean, dry gunny bags away from moisture."
    ),
    "soil": (
        "Cotton grows best in black cotton soils (vertisols) and alluvial soils. "
        "Ideal pH: 6.0-8.0. Deep ploughing before monsoon, followed by 2-3 harrowings. "
        "Add FYM or compost (10 tonnes/ha) during land preparation. "
        "Ensure proper drainage to prevent waterlogging."
    ),
}

# Hindi knowledge base
KNOWLEDGE_BASE_HI = {
    "bacterial blight": (
        "जीवाणु झुलसा Xanthomonas citri subsp. malvacearum जीवाणु के कारण होता है। "
        "पत्तियों पर कोणीय पान से भरे घाव दिखते हैं। रोग मुक्त बीज, फसल चक्र और "
        "तांबा आधारित स्प्रे से रोकथाम करें। स्ट्रेप्टोमाइसिन सल्फेट से उपचार करें।"
    ),
    "curl virus": (
        "कपास पत्ती कर्ल वायरस सफेद मक्खी द्वारा फैलता है। पत्तियां मुड़ती और मोटी हो जाती हैं। "
        "वायरस का सीधा इलाज नहीं है। सफेद मक्खी नियंत्रण के लिए Imidacloprid का उपयोग करें।"
    ),
    "fusarium wilt": (
        "फ्यूज़ेरियम विल्ट मिट्टी जनित कवक रोग है। निचली पत्तियां पीली होकर मुरझाती हैं। "
        "प्रतिरोधी किस्में लगाएं, मिट्टी pH 6.0-7.0 रखें, 4-5 वर्ष फसल चक्र अपनाएं।"
    ),
    "irrigation": (
        "कपास को 500-700 mm पानी चाहिए। फूल आने और बॉल बनने के समय सिंचाई महत्वपूर्ण है। "
        "ड्रिप सिंचाई जल दक्षता के लिए सर्वोत्तम है। जलभराव से बचें।"
    ),
    "fertilizer": (
        "अनुशंसित NPK: 100-150 kg N, 50-60 kg P2O5, 50-60 kg K2O प्रति हेक्टेयर। "
        "नाइट्रोजन विभाजित मात्रा में दें। मिट्टी परीक्षण अवश्य करें।"
    ),
    "pest": (
        "प्रमुख कीट: बॉलवर्म, सफेद मक्खी, एफिड्स। एकीकृत कीट प्रबंधन अपनाएं। "
        "नीम तेल 5% स्प्रे करें। क्षेत्र स्वच्छता बनाए रखें।"
    ),
    "farming practices": (
        "सर्वोत्तम कपास खेती: प्रमाणित बीज, उचित दूरी, संतुलित उर्वरक, "
        "नियमित कीट निगरानी, समय पर सिंचाई, 80% बॉल खुलने पर कटाई।"
    ),
    "crop maintenance": (
        "पतला करना 15-20 दिन बाद। निराई 20-25 और 45-50 DAS पर। "
        "रोगग्रस्त पौधे तुरंत हटाएं। साप्ताहिक कीट निगरानी करें।"
    ),
}

# Marathi knowledge base
KNOWLEDGE_BASE_MR = {
    "bacterial blight": (
        "जीवाणू झुलसा Xanthomonas जीवाणूमुळे होतो. पानांवर कोनिय जखमे दिसतात. "
        "रोगमुक्त बियाणे, पीक फेरपालट आणि तांबे आधारित फवारणी करा."
    ),
    "curl virus": (
        "कापूस पान कर्ल व्हायरस पांढरी माशीद्वारे पसरतो. पाने वाकतात आणि जाड होतात. "
        "व्हायरसचा थेट उपचार नाही. पांढरी माशी नियंत्रणासाठी Imidacloprid वापरा."
    ),
    "fusarium wilt": (
        "फ्युजेरियम विल्ट मातीजन्य बुरशीजन्य रोग आहे. खालची पाने पिवळी होतात. "
        "प्रतिरोधक जाती लावा, माती pH 6.0-7.0 ठेवा."
    ),
    "irrigation": (
        "कापूसाला 500-700 mm पाणी लागते. फुलण्याच्या वेळी सिंचन महत्त्वाचे आहे. "
        "ड्रिप सिंचन पाणी बचतीसाठी सर्वोत्तम आहे."
    ),
    "fertilizer": (
        "शिफारस केलेले NPK: 100-150 kg N, 50-60 kg P2O5, 50-60 kg K2O प्रति हेक्टर. "
        "माती चाचणी अवश्य करा."
    ),
    "pest": (
        "प्रमुख कीड: बॉलवर्म, पांढरी माशी. एकात्मिक कीड व्यवस्थापन करा. "
        "कडुलिंब तेल 5% फवारणी करा."
    ),
    "farming practices": (
        "सर्वोत्तम कापूस शेती: प्रमाणित बियाणे, योग्य अंतर, संतुलित खते, "
        "नियमित कीड देखरेख, वेळेवर सिंचन."
    ),
    "crop maintenance": (
        "१५-२० दिवसांनी विरळ करा. रोगग्रस्त झाडे त्वरित काढा. "
        "साप्ताहिक कीड देखरेख करा."
    ),
}

KNOWLEDGE_BASES = {
    "en": KNOWLEDGE_BASE_EN,
    "hi": KNOWLEDGE_BASE_HI,
    "mr": KNOWLEDGE_BASE_MR,
}

# Expanded Intent Training Data for ML-based NLP matching
INTENT_EXAMPLES = {
    "bacterial blight": [
        "what is bacterial blight", "tell me about bacterial blight", "symptoms of bacterial blight", 
        "xanthomonas infection", "angular water-soaked lesions on leaves", "brown spots on leaves",
        "how to treat bacterial blight", "जीवाणु झुलसा क्या है", "जीवाणू झुलसा", "बॅक्टेरियल ब्लाइट",
        "my plant has bacterial blight", "copper oxychloride use"
    ],
    "curl virus": [
        "what is curl virus", "cotton leaf curl virus", "clcuv", "leaves are curling upward", 
        "thickening of leaves", "whitefly virus", "how to cure leaf curl", "पत्ती कर्ल वायरस",
        "पान कर्ल व्हायरस", "पत्ते मुड़ रहे हैं", "leaves folding"
    ],
    "fusarium wilt": [
        "what is fusarium wilt", "fusarium oxysporum", "leaves are yellowing and wilting",
        "plant is wilting during hot hours", "how to prevent fusarium wilt", "फ्यूज़ेरियम विल्ट",
        "फ्युजेरियम विल्ट", "पौधा सूख रहा है", "झाड वाळत आहे"
    ],
    "healthy": [
        "is my plant healthy", "how to keep plant healthy", "healthy cotton plant",
        "no disease", "स्वस्थ पौधा", "निरोगी झाड", "good condition plant"
    ],
    "irrigation": [
        "how much water to give", "irrigation schedule", "when to water cotton",
        "water requirement", "drip irrigation", "सिंचाई कब करें", "सिंचन कसे करावे",
        "पानी कितना दें", "पाणी कधी द्यावे"
    ],
    "fertilizer": [
        "which fertilizer to use", "npk ratio for cotton", "nutrient management",
        "how much urea", "fertiliser recommendations", "उर्वरक कौन सा डालें", "खत कोणते द्यावे",
        "खाद की जानकारी", "fym requirement"
    ],
    "pest": [
        "how to control pests", "insect attack", "bollworm problem", "whitefly control",
        "aphids and jassids", "mealybugs", "neem oil spray", "कीट नियंत्रण", "कीड व्यवस्थापन",
        "बोंडअळी", "पांढरी माशी"
    ],
    "farming practices": [
        "best farming practices", "how to grow cotton", "cotton cultivation tips",
        "spacing for cotton", "yield improvement", "कपास की खेती कैसे करें", "कापूस शेती पद्धती",
        "अधिक उपज"
    ],
    "crop maintenance": [
        "how to maintain crop", "weeding schedule", "thinning of plants", "crop care",
        "pruning cotton", "रखरखाव कैसे करें", "पीक देखभाल", "निराई गुड़ाई", "तण नियंत्रण"
    ],
    "sowing": [
        "when to sow cotton", "sowing time", "seed rate", "planting depth",
        "kharif season sowing", "बुवाई का समय", "पेरणी कधी करावी", "बियाणे किती लागते"
    ],
    "harvest": [
        "when to harvest", "cotton picking", "boll opening", "harvesting time",
        "कटाई कब करें", "कापणी कधी करावी", "वेचणी"
    ],
    "soil": [
        "best soil for cotton", "soil ph", "black cotton soil", "land preparation",
        "मिट्टी कैसी होनी चाहिए", "माती कशी असावी", "जमीन"
    ],
}

# Initialize TF-IDF model if sklearn is available
if SKLEARN_AVAILABLE:
    intent_labels = []
    corpus = []
    for intent, examples in INTENT_EXAMPLES.items():
        for ex in examples:
            corpus.append(ex.lower())
            intent_labels.append(intent)
            
    # Using ngram_range (1,2) to capture context better and removing common english stop words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)


def _match_intent(user_message: str) -> Optional[str]:
    """Match user message to the most relevant knowledge base topic using NLP / ML matching."""
    if not user_message or not user_message.strip():
        return None
        
    message_lower = user_message.lower().strip()
    
    if SKLEARN_AVAILABLE:
        # ML-based matching with Cosine Similarity
        user_vec = vectorizer.transform([message_lower])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        
        # Threshold ensures we don't match completely random sentences
        if best_score > 0.08:
            return intent_labels[best_idx]
            
    # Fallback heuristic matching if ML score is too low or sklearn unavailable
    best_match = None
    best_score = 0

    for intent, examples in INTENT_EXAMPLES.items():
        score = 0
        for ex in examples:
            # Check for exact substring match in examples
            if ex in message_lower:
                score += 5
            
        if score > best_score:
            best_score = score
            best_match = intent

    return best_match if best_score >= 5 else None


def get_rule_based_response(user_message: str, lang: str = "en") -> str:
    """
    Generate a rule-based response from the knowledge base.

    Args:
        user_message: User's chat message.
        lang: Language code ('en', 'hi', 'mr').

    Returns:
        Response string.
    """
    kb = KNOWLEDGE_BASES.get(lang, KNOWLEDGE_BASE_EN)
    intent = _match_intent(user_message)

    if intent and intent in kb:
        return kb[intent]

    # Fallback to English knowledge base if not found in target language
    if lang != "en":
        intent = _match_intent(user_message)
        if intent and intent in KNOWLEDGE_BASE_EN:
            return KNOWLEDGE_BASE_EN[intent]

    # Default responses
    defaults = {
        "en": (
            "I'm your Cotton Farming Assistant! I can help with:\n"
            "- Disease information (Bacterial Blight, Curl Virus, Fusarium Wilt)\n"
            "- Irrigation and fertilizer advice\n"
            "- Pest management and crop maintenance\n"
            "- Best cotton farming practices\n\n"
            "Please ask a specific question about cotton farming."
        ),
        "hi": (
            "मैं आपका कपास खेती सहायक हूं! मैं इन विषयों में मदद कर सकता हूं:\n"
            "- रोग जानकारी (जीवाणु झुलसा, कर्ल वायरस, फ्यूज़ेरियम विल्ट)\n"
            "- सिंचाई और उर्वरक सलाह\n"
            "- कीट प्रबंधन और फसल रखरखाव\n"
            "- सर्वोत्तम कपास खेती प्रथाएं"
        ),
        "mr": (
            "मी तुमचा कापूस शेती सहाय्यक आहे! मी या विषयांमध्ये मदत करू शकतो:\n"
            "- रोग माहिती (जीवाणू झुलसा, कर्ल व्हायरस, फ्युजेरियम विल्ट)\n"
            "- सिंचन आणि खते सल्ला\n"
            "- कीड व्यवस्थापन आणि पीक देखभाल\n"
            "- सर्वोत्तम कापूस शेती पद्धती"
        ),
    }
    return defaults.get(lang, defaults["en"])


def is_gemini_available() -> bool:
    """Check if Gemini API key is configured."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or getattr(config, "GEMINI_API_KEY", "")
    return bool(api_key)


def get_gemini_response(user_message: str, lang: str = "en") -> str:
    """
    Get response from Google Gemini API.

    Args:
        user_message: User's chat message.
        lang: Language code for response language.

    Returns:
        Gemini-generated response or fallback to rule-based.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or getattr(config, "GEMINI_API_KEY", "")
    if not api_key:
        return get_rule_based_response(user_message, lang)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        lang_names = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
        lang_name = lang_names.get(lang, "English")

        system_prompt = (
            f"You are a Smart Cotton Farming Assistant specializing in cotton plant diseases "
            f"and agricultural advisory. Answer questions about cotton farming, diseases "
            f"(Bacterial Blight, Curl Virus, Fusarium Wilt), irrigation, fertilizers, "
            f"pest management, and crop maintenance. Respond in {lang_name}. "
            f"Be concise, practical, and helpful to farmers."
        )

        response = model.generate_content(
            f"{system_prompt}\n\nUser question: {user_message}"
        )
        return response.text

    except Exception as e:
        import traceback
        print("Gemini API Exception:")
        traceback.print_exc()
        return get_rule_based_response(user_message, lang)


def get_chatbot_response(user_message: str, lang: str = "en") -> Tuple[str, str]:
    """
    Get chatbot response using Gemini if available, otherwise rule-based.

    Args:
        user_message: User's chat message.
        lang: Language code.

    Returns:
        Tuple of (response_text, mode_used).
    """
    if is_gemini_available():
        response = get_gemini_response(user_message, lang)
        return response, "Gemini AI"
    else:
        response = get_rule_based_response(user_message, lang)
        return response, "Rule-Based Knowledge"
