"""
Disease information system for cotton plant diseases.
Stores symptoms, causes, prevention, treatment, and yield impact
with multilingual support (English, Hindi, Marathi).
"""

from translations import translate_disease_name

# Disease knowledge base with multilingual content
DISEASE_DATABASE = {
    "Healthy": {
        "en": {
            "name": "Healthy",
            "symptoms": (
                "Uniform green color across the leaf surface. "
                "No spots, lesions, curling, or discoloration. "
                "Firm leaf texture with normal venation pattern. "
                "No wilting or yellowing observed."
            ),
            "causes": (
                "Proper irrigation and balanced fertilization. "
                "Adequate sunlight and good soil drainage. "
                "Effective pest and disease management practices. "
                "Use of disease-resistant cotton varieties."
            ),
            "prevention": (
                "Maintain regular field monitoring schedules. "
                "Apply balanced NPK fertilizers as per soil test. "
                "Ensure proper spacing between plants for airflow. "
                "Practice crop rotation with non-host crops."
            ),
            "treatment": (
                "Continue current good agricultural practices. "
                "Monitor plants weekly for early disease signs. "
                "Maintain soil moisture at optimal levels. "
                "Apply micronutrients if deficiency symptoms appear."
            ),
            "impact_on_yield": (
                "Healthy plants can achieve 100% of their yield potential. "
                "Proper maintenance ensures high fiber quality and boll count. "
                "No yield loss expected when plants remain disease-free."
            ),
        },
        "hi": {
            "name": "स्वस्थ",
            "symptoms": (
                "पत्ती की सतह पर समान हरा रंग। "
                "कोई धब्बे, घाव, मुड़ना या रंग परिवर्तन नहीं। "
                "सामान्य शिरा पैटर्न के साथ मजबूत पत्ती बनावट। "
                "कोई मुरझाना या पीला पड़ना नहीं देखा गया।"
            ),
            "causes": (
                "उचित सिंचाई और संतुलित उर्वरक। "
                "पर्याप्त धूप और अच्छी मिट्टी जल निकासी। "
                "प्रभावी कीट और रोग प्रबंधन। "
                "रोग प्रतिरोधी कपास किस्मों का उपयोग।"
            ),
            "prevention": (
                "नियमित खेत निगरानी अनुसूची बनाए रखें। "
                "मिट्टी परीक्षण के अनुसार संतुलित NPK उर्वरक लगाएं। "
                "हवा के लिए पौधों के बीच उचित दूरी सुनिश्चित करें। "
                "गैर-मेजबान फसलों के साथ फसल चक्र अपनाएं।"
            ),
            "treatment": (
                "वर्तमान अच्छी कृषि प्रथाओं को जारी रखें। "
                "प्रारंभिक रोग संकेतों के लिए साप्ताहिक निगरानी करें। "
                "इष्टतम स्तर पर मिट्टी की नमी बनाए रखें। "
                "कमी लक्षण दिखने पर सूक्ष्म पोषक तत्व लगाएं।"
            ),
            "impact_on_yield": (
                "स्वस्थ पौधे अपनी उपज क्षमता का 100% प्राप्त कर सकते हैं। "
                "उचित रखरखाव उच्च रेशा गुणवत्ता और बॉल संख्या सुनिश्चित करता है। "
                "रोग मुक्त रहने पर कोई उपज हानि नहीं।"
            ),
        },
        "mr": {
            "name": "निरोगी",
            "symptoms": (
                "पानाच्या पृष्ठभागावर समान हिरवा रंग. "
                "कोणतेही डाग, जखमा, वाकणे किंवा रंग बदल नाही. "
                "सामान्य शिरा नमुन्यासह मजबूत पान बनावट. "
                "कोणतेही वाळणे किंवा पिवळेपणा दिसत नाही."
            ),
            "causes": (
                "योग्य सिंचन आणि संतुलित खत. "
                "पुरेशी सूर्यप्रकाश आणि चांगली माती जलनिकास. "
                "प्रभावी कीड आणि रोग व्यवस्थापन. "
                "रोग प्रतिरोधक कापूस जातींचा वापर."
            ),
            "prevention": (
                "नियमित शेत देखरेखीचे वेळापत्रक राखा. "
                "माती चाचणीनुसार संतुलित NPK खते वापरा. "
                "हवेच्या प्रवाहासाठी झाडांमध्ये योग्य अंतर ठेवा. "
                "गैर-यजमान पिकांसह पीक फेरपालट करा."
            ),
            "treatment": (
                "सध्याच्या चांगल्या शेती पद्धती सुरू ठेवा. "
                "लवकर रोग लक्षणांसाठी साप्ताहिक देखरेख करा. "
                "इष्टतम पातळीवर मातीची ओलसरता राखा. "
                "कमतरता लक्षणे दिसल्यास सूक्ष्म पोषक तत्वे द्या."
            ),
            "impact_on_yield": (
                "निरोगी रोपे त्यांच्या उत्पादन क्षमतेचे 100% साध्य करू शकतात. "
                "योग्य देखभाल उच्च तंतू गुणवत्ता आणि बॉल संख्या सुनिश्चित करते. "
                "रोगमुक्त राहिल्यास कोणतेही उत्पादन नुकसान नाही."
            ),
        },
    },
    "Bacterial Blight": {
        "en": {
            "name": "Bacterial Blight",
            "symptoms": (
                "Angular water-soaked lesions on leaves bordered by leaf veins. "
                "Lesions turn brown to black with a yellow halo. "
                "Premature leaf drop and boll rot in severe cases. "
                "Dark streaks on stems and petioles may appear."
            ),
            "causes": (
                "Caused by Xanthomonas citri subsp. malvacearum bacteria. "
                "Spreads through rain splash, irrigation water, and farm tools. "
                "Favored by warm temperatures (25-30°C) and high humidity. "
                "Infected seeds and crop debris serve as primary inoculum."
            ),
            "prevention": (
                "Use certified disease-free seeds. "
                "Practice crop rotation for at least 2-3 years. "
                "Avoid overhead irrigation to reduce leaf wetness. "
                "Remove and destroy infected plant debris after harvest. "
                "Apply copper-based bactericides as preventive spray."
            ),
            "treatment": (
                "Spray Streptomycin sulphate (200 ppm) or Copper oxychloride. "
                "Apply bactericides at first sign of infection. "
                "Repeat sprays at 10-15 day intervals during epidemic. "
                "Remove severely infected plants to prevent spread. "
                "Improve field drainage and reduce plant density."
            ),
            "impact_on_yield": (
                "Can cause 10-50% yield loss depending on severity. "
                "Boll rot reduces fiber quality significantly. "
                "Early infection during flowering stage causes maximum damage. "
                "Severe epidemics may require complete field replanting."
            ),
        },
        "hi": {
            "name": "जीवाणु झुलसा",
            "symptoms": (
                "पत्ती शिराओं से घिरे कोणीय पान से भरे घाव। "
                "घाव पीले प्रभामंडल के साथ भूरे से काले हो जाते हैं। "
                "गंभीर मामलों में समय से पहले पत्ती गिरना और बॉल सड़ना। "
                "तने और पत्ती डंडियों पर काली धारियां दिख सकती हैं।"
            ),
            "causes": (
                "Xanthomonas citri subsp. malvacearum जीवाणु के कारण। "
                "बारिश, सिंचाई पानी और कृषि औजारों से फैलता है। "
                "गर्म तापमान (25-30°C) और उच्च आर्द्रता में बढ़ता है। "
                "संक्रमित बीज और फसल अवशेष प्राथमिक स्रोत हैं।"
            ),
            "prevention": (
                "प्रमाणित रोग मुक्त बीज का उपयोग करें। "
                "कम से कम 2-3 वर्ष की फसल चक्र अपनाएं। "
                "पत्ती की नमी कम करने के लिए ओवरहेड सिंचाई से बचें। "
                "कटाई के बाद संक्रमित पौधे अवशेष हटाएं और नष्ट करें। "
                "तांबा आधारित जीवाणुनाशक निवारक स्प्रे लगाएं।"
            ),
            "treatment": (
                "स्ट्रेप्टोमाइसिन सल्फेट (200 ppm) या कॉपर ऑक्सीक्लोराइड स्प्रे करें। "
                "संक्रमण के पहले संकेत पर जीवाणुनाशक लगाएं। "
                "महामारी के दौरान 10-15 दिन के अंतराल पर दोहराएं। "
                "फैलाव रोकने के लिए गंभीर रूप से संक्रमित पौधे हटाएं। "
                "खेत जल निकासी में सुधार करें और पौधे घनत्व कम करें।"
            ),
            "impact_on_yield": (
                "गंभीरता के आधार पर 10-50% उपज हानि हो सकती है। "
                "बॉल सड़न से रेशा गुणवत्ता काफी कम हो जाती है। "
                "फूल आने के चरण में प्रारंभिक संक्रमण अधिकतम नुकसान करता है। "
                "गंभीर महामारी में पूरे खेत की दोबारा बुवाई आवश्यक हो सकती है।"
            ),
        },
        "mr": {
            "name": "जीवाणू झुलसा",
            "symptoms": (
                "पानाच्या शिरांनी वेढलेले कोनिय पाण्याने भरलेले जखमे. "
                "जखमे पिवळ्या प्रभामंडळासह तपकिरी ते काळे होतात. "
                "गंभीर प्रकरणांमध्ये लवकर पाने गळणे आणि बॉल कुजणे. "
                "खोड आणि पानाच्या देऱ्यावर काळ्या रेषा दिसू शकतात."
            ),
            "causes": (
                "Xanthomonas citri subsp. malvacearum जीवाणूमुळे. "
                "पाऊस, सिंचन पाणी आणि शेती साधने यांद्वारे पसरते. "
                "उबदार तापमान (25-30°C) आणि उच्च आर्द्रतेत वाढते. "
                "संक्रमित बियाणे आणि पीक अवशेष प्राथमिक स्रोत आहेत."
            ),
            "prevention": (
                "प्रमाणित रोगमुक्त बियाणे वापरा. "
                "किमान 2-3 वर्षांची पीक फेरपालट करा. "
                "पानांची ओलसरता कमी करण्यासाठी वरून सिंचन टाळा. "
                "कापणीनंतर संक्रमित वनस्पती अवशेष काढून नष्ट करा. "
                "तांबे आधारित जीवाणुनाशक प्रतिबंधात्मक फवारणी करा."
            ),
            "treatment": (
                "स्ट्रेप्टोमायसिन सल्फेट (200 ppm) किंवा कॉपर ऑक्सीक्लोराइड फवारा. "
                "संक्रमणाच्या पहिल्या चिन्हावर जीवाणुनाशक लावा. "
                "महामारी दरम्यान 10-15 दिवसांच्या अंतराने पुन्हा करा. "
                "प्रसार रोखण्यासाठी गंभीर संक्रमित झाडे काढा. "
                "शेत जलनिकास सुधारा आणि झाडांची घनता कमी करा."
            ),
            "impact_on_yield": (
                "तीव्रतेनुसार 10-50% उत्पादन नुकसान होऊ शकते. "
                "बॉल कुजण्यामुळे तंतू गुणवत्ता लक्षणीयरीत्या कमी होते. "
                "फुलण्याच्या टप्प्यात लवकर संक्रमण जास्त नुकसान करते. "
                "गंभीर महामारीत संपूर्ण शेत पुन्हा लावणे आवश्यक असू शकते."
            ),
        },
    },
    "Curl Virus": {
        "en": {
            "name": "Curl Virus",
            "symptoms": (
                "Upward curling and crinkling of young leaves. "
                "Thickening and dark green discoloration of leaves. "
                "Stunted plant growth with shortened internodes. "
                "Reduced boll formation and smaller boll size. "
                "Leaf veins may appear thickened and distorted."
            ),
            "causes": (
                "Caused by Cotton Leaf Curl Virus (CLCuV) transmitted by whiteflies. "
                "Bemisia tabaci whitefly is the primary vector. "
                "Weeds like Abutilon and Malvastrum act as reservoir hosts. "
                "Warm dry conditions favor whitefly population buildup."
            ),
            "prevention": (
                "Grow CLCuV-resistant cotton varieties (e.g., PKV 081). "
                "Install yellow sticky traps for whitefly monitoring. "
                "Remove weed hosts from field borders regularly. "
                "Use reflective mulches to deter whitefly landing. "
                "Avoid planting near infected fields."
            ),
            "treatment": (
                "No direct cure exists for viral infections. "
                "Control whitefly vectors using Imidacloprid or Thiamethoxam. "
                "Apply neem oil sprays (5%) at weekly intervals. "
                "Uproot and destroy virus-infected plants immediately. "
                "Boost plant immunity with potassium-rich fertilizers."
            ),
            "impact_on_yield": (
                "Can reduce yield by 30-70% in severe infections. "
                "Fiber quality deteriorates with shortened staple length. "
                "Early season infection causes complete crop failure. "
                "Economic losses are highest in susceptible varieties."
            ),
        },
        "hi": {
            "name": "कर्ल वायरस",
            "symptoms": (
                "युवा पत्तियों का ऊपर की ओर मुड़ना और सिकुड़ना। "
                "पत्तियों का मोटा होना और गहरा हरा रंग परिवर्तन। "
                "छोटे अंतराल वाली रुकी हुई पौधे वृद्धि। "
                "बॉल निर्माण में कमी और छोटा बॉल आकार। "
                "पत्ती शिराए मोटी और विकृत दिख सकती हैं।"
            ),
            "causes": (
                "सफेद मक्खी द्वारा संचारित कपास पत्ती कर्ल वायरस (CLCuV) के कारण। "
                "Bemisia tabaci सफेद मक्खी प्राथमिक वाहक है। "
                "Abutilon और Malvastrum जैसी खरपतवार रिजर्वोयर मेजबान हैं। "
                "गर्म शुष्क परिस्थितियां सफेद मक्खी आबादी बढ़ाने में सहायक हैं।"
            ),
            "prevention": (
                "CLCuV प्रतिरोधी कपास किस्में उगाएं (जैसे PKV 081)। "
                "सफेद मक्खी निगरानी के लिए पीले चिपचिपे जाल लगाएं। "
                "खेत की सीमाओं से खरपतवार मेजबान नियमित हटाएं। "
                "सफेद मक्खी को रोकने के लिए परावर्तक मल्च का उपयोग करें। "
                "संक्रमित खेतों के पास बुवाई से बचें।"
            ),
            "treatment": (
                "वायरल संक्रमण के लिए कोई प्रत्यक्ष इलाज नहीं है। "
                "Imidacloprid या Thiamethoxam से सफेद मक्खी वाहक नियंत्रण करें। "
                "साप्ताहिक अंतराल पर नीम तेल स्प्रे (5%) लगाएं। "
                "वायरस संक्रमित पौधे तुरंत उखाड़ें और नष्ट करें। "
                "पोटेशियम युक्त उर्वरकों से पौधे की प्रतिरक्षा बढ़ाएं।"
            ),
            "impact_on_yield": (
                "गंभीर संक्रमण में 30-70% उपज कमी हो सकती है। "
                "छोटी स्टेपल लंबाई के साथ रेशा गुणवत्ता खराब होती है। "
                "प्रारंभिक सीजन संक्रमण से पूरी फसल विफलता हो सकती है। "
                "संवेदनशील किस्मों में आर्थिक नुकसान सबसे अधिक होता है।"
            ),
        },
        "mr": {
            "name": "कर्ल व्हायरस",
            "symptoms": (
                "तरुण पानांचे वरच्या दिशेने वाकणे आणि कुरतडणे. "
                "पानांचे जाड होणे आणि गडद हिरवा रंग बदल. "
                "लहान अंतरासह अवरुद्ध झाड वाढ. "
                "बॉल निर्मिती कमी आणि लहान बॉल आकार. "
                "पानाच्या शिरा जाड आणि विकृत दिसू शकतात."
            ),
            "causes": (
                "पांढरी माशीद्वारे संचारित कापूस पान कर्ल व्हायरस (CLCuV) मुळे. "
                "Bemisia tabaci पांढरी माशी प्राथमिक वाहक आहे. "
                "Abutilon आणि Malvastrum सारख्या तण यजमान म्हणून काम करतात. "
                "उबदार कोरड्या परिस्थितीत पांढरी माशी लोकसंख्या वाढते."
            ),
            "prevention": (
                "CLCuV प्रतिरोधक कापूस जाती लावा (उदा. PKV 081). "
                "पांढरी माशी देखरेखीसाठी पिवळे चिकट सापळे लावा. "
                "शेताच्या सीमांवरून तण यजमान नियमित काढा. "
                "पांढरी माशी रोखण्यासाठी परावर्तक मल्च वापरा. "
                "संक्रमित शेतांच्या जवळ लावणी टाळा."
            ),
            "treatment": (
                "व्हायरल संक्रमणासाठी थेट उपचार नाही. "
                "Imidacloprid किंवा Thiamethoxam ने पांढरी माशी वाहक नियंत्रण करा. "
                "साप्ताहिक अंतराने कडुलिंब तेल फवारणी (5%) करा. "
                "व्हायरस संक्रमित झाडे त्वरित उपटून नष्ट करा. "
                "पोटॅशियम समृद्ध खते झाडांची प्रतिकारशक्ती वाढवा."
            ),
            "impact_on_yield": (
                "गंभीर संक्रमणात 30-70% उत्पादन कमी होऊ शकते. "
                "लहान स्टेपल लांबीसह तंतू गुणवत्ता खराब होते. "
                "लवकर हंगामातील संक्रमणामुळे संपूर्ण पीक अयशस्वी होऊ शकते. "
                "संवेदनशील जातींमध्ये आर्थिक नुकसान सर्वाधिक असते."
            ),
        },
    },
    "Fusarium Wilt": {
        "en": {
            "name": "Fusarium Wilt",
            "symptoms": (
                "Yellowing of lower leaves progressing upward. "
                "Wilting during hottest part of the day, partial recovery at night. "
                "Brown discoloration of vascular tissues when stem is cut. "
                "Stunted growth with only a few bolls produced. "
                "Plants may die completely in severe infections."
            ),
            "causes": (
                "Caused by Fusarium oxysporum f. sp. vasinfectum fungus. "
                "Soil-borne pathogen that persists for many years. "
                "Spreads through contaminated soil, water, and equipment. "
                "Favored by acidic soils and temperatures of 25-32°C. "
                "Nematode damage increases susceptibility."
            ),
            "prevention": (
                "Plant Fusarium wilt-resistant varieties. "
                "Practice long crop rotation (4-5 years) with cereals. "
                "Maintain soil pH between 6.0 and 7.0 with lime application. "
                "Avoid waterlogging and improve soil drainage. "
                "Treat seeds with Trichoderma viride bio-agent."
            ),
            "treatment": (
                "Apply Carbendazim or Thiophanate methyl as soil drench. "
                "Use biological control with Pseudomonas fluorescens. "
                "Remove and burn wilted plants to reduce inoculum. "
                "Solarize soil during summer fallow period. "
                "Apply organic matter to improve soil health."
            ),
            "impact_on_yield": (
                "Can cause 20-80% yield loss depending on race and variety. "
                "Infected plants produce fewer and smaller bolls. "
                "Soil infestation affects future cotton crops for years. "
                "Race 4 Fusarium can devastate entire fields."
            ),
        },
        "hi": {
            "name": "फ्यूज़ेरियम विल्ट",
            "symptoms": (
                "निचले पत्तों का पीला होना ऊपर की ओर बढ़ता है। "
                "दिन के सबसे गर्म समय में मुरझाना, रात में आंशिक восстановление। "
                "तना काटने पर संवहनी ऊतकों का भूरा रंग परिवर्तन। "
                "कुछ बॉल के साथ रुकी हुई वृद्धि। "
                "गंभीर संक्रमण में पौधे पूरी तरह मर सकते हैं।"
            ),
            "causes": (
                "Fusarium oxysporum f. sp. vasinfectum कवक के कारण। "
                "मिट्टी जनित रोगजनक जो कई वर्षों तक रहता है। "
                "दूषित मिट्टी, पानी और उपकरणों से फैलता है। "
                "अम्लीय मिट्टी और 25-32°C तापमान में बढ़ता है। "
                "नेमाटोड क्षति संवेदनशीलता बढ़ाती है।"
            ),
            "prevention": (
                "फ्यूज़ेरियम विल्ट प्रतिरोधी किस्में लगाएं। "
                "अनाज के साथ लंबी फसल चक्र (4-5 वर्ष) अपनाएं। "
                "चूने के उपयोग से मिट्टी pH 6.0-7.0 बनाए रखें। "
                "जलभराव से बचें और मिट्टी जल निकासी सुधारें। "
                "Trichoderma viride जैव एजेंट से बीज उपचार करें।"
            ),
            "treatment": (
                "Carbendazim या Thiophanate methyl मिट्टी ड्रेंच के रूप में लगाएं। "
                "Pseudomonas fluorescens के साथ जैविक नियंत्रण करें। "
                "रोगजनक कम करने के लिए मुरझाए पौधे हटाकर जलाएं। "
                "गर्मी की परती अवधि के दौरान मिट्टी सौरकरण करें। "
                "मिट्टी स्वास्थ्य सुधारने के लिए जैविक पदार्थ लगाएं।"
            ),
            "impact_on_yield": (
                "जाति और किस्म के आधार पर 20-80% उपज हानि हो सकती है। "
                "संक्रमित पौधे कम और छोटे बॉल उत्पादन करते हैं। "
                "मिट्टी संक्रमण वर्षों तक भविष्य की कपास फसलों को प्रभावित करता है। "
                "रेस 4 फ्यूज़ेरियम पूरे खेतों को तबाह कर सकता है।"
            ),
        },
        "mr": {
            "name": "फ्युजेरियम विल्ट",
            "symptoms": (
                "खालच्या पानांचा पिवळा होणे वरच्या दिशेने वाढते. "
                "दिवसातील सर्वात उबदार वेळी वाळणे, रात्री आंशिक पुनर्प्राप्ती. "
                "खोड कापल्यावर वाहिनी ऊतकांचा तपकिरी रंग बदल. "
                "काही बॉलसह अवरुद्ध वाढ. "
                "गंभीर संक्रमणात झाडे पूर्णपणे मरू शकतात."
            ),
            "causes": (
                "Fusarium oxysporum f. sp. vasinfectum बुरशीमुळे. "
                "मातीजन्य रोगजनक जो अनेक वर्षे टिकून राहतो. "
                "दूषित माती, पाणी आणि उपकरणांद्वारे पसरतो. "
                "आम्लीय माती आणि 25-32°C तापमानात वाढते. "
                "नेमाटोड नुकसान संवेदनशीलता वाढवते."
            ),
            "prevention": (
                "फ्युजेरियम विल्ट प्रतिरोधक जाती लावा. "
                "धान्यांसह दीर्घ पीक फेरपालट (4-5 वर्ष) करा. "
                "चुना वापरून माती pH 6.0-7.0 राखा. "
                "पाणी साचू देऊ नका आणि माती जलनिकास सुधारा. "
                "Trichoderma viride जैव एजंटने बियाणे प्रक्रिया करा."
            ),
            "treatment": (
                "Carbendazim किंवा Thiophanate methyl माती ड्रेंच म्हणून लावा. "
                "Pseudomonas fluorescens सह जैविक नियंत्रण करा. "
                "रोगजनक कमी करण्यासाठी वाळलेली झाडे काढून जाळा. "
                "उन्हाळ्यातील पडी काळात माती सौरकरण करा. "
                "माती आरोग्य सुधारण्यासाठी सेंद्रिय पदार्थ लावा."
            ),
            "impact_on_yield": (
                "जाती आणि वाणानुसार 20-80% उत्पादन नुकसान होऊ शकते. "
                "संक्रमित झाडे कमी आणि लहान बॉल तयार करतात. "
                "माती संक्रमण वर्षानुवर्षे भविष्यातील कापूस पिकांवर परिणाम करते. "
                "रेस 4 फ्युजेरियम संपूर्ण शेत नष्ट करू शकते."
            ),
        },
    },
}


def get_disease_info(disease_name: str, lang: str = "en") -> dict:
    """
    Retrieve disease information for a given disease and language.

    Args:
        disease_name: One of the CLASS_NAMES disease labels.
        lang: Language code ('en', 'hi', 'mr').

    Returns:
        Dictionary with keys: name, symptoms, causes, prevention,
        treatment, impact_on_yield.
    """
    if disease_name not in DISEASE_DATABASE:
        return {
            "name": translate_disease_name(disease_name, lang),
            "symptoms": "Information not available.",
            "causes": "Information not available.",
            "prevention": "Information not available.",
            "treatment": "Information not available.",
            "impact_on_yield": "Information not available.",
        }

    lang_data = DISEASE_DATABASE[disease_name].get(lang)
    if lang_data is None:
        lang_data = DISEASE_DATABASE[disease_name]["en"]

    return lang_data


def get_all_disease_names(lang: str = "en") -> list:
    """Return list of all disease names in the specified language."""
    return [
        DISEASE_DATABASE[d].get(lang, DISEASE_DATABASE[d]["en"])["name"]
        for d in DISEASE_DATABASE
    ]
