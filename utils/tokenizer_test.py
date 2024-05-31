from tokenizer import expand_numbers_multilingual, expand_abbreviations_multilingual, expand_symbols_multilingual

def test_expand_numbers_multilingual():
    test_cases = [
        # English
        ("In 12.5 seconds.", "In twelve point five seconds.", "en"),
        ("There were 50 soldiers.", "There were fifty soldiers.", "en"),
        ("This is a 1st test", "This is a first test", "en"),
        ("That will be $20 sir.", "That will be twenty dollars sir.", "en"),
        ("That will be 20€ sir.", "That will be twenty euro sir.", "en"),
        (
            "That will be 20.15€ sir.",
            "That will be twenty euro, fifteen cents sir.",
            "en",
        ),
        ("That's 100,000.5.", "That's one hundred thousand point five.", "en"),
        # French
        ("En 12,5 secondes.", "En douze virgule cinq secondes.", "fr"),
        ("Il y avait 50 soldats.", "Il y avait cinquante soldats.", "fr"),
        ("Ceci est un 1er test", "Ceci est un premier test", "fr"),
        (
            "Cela vous fera $20 monsieur.",
            "Cela vous fera vingt dollars monsieur.",
            "fr",
        ),
        ("Cela vous fera 20€ monsieur.", "Cela vous fera vingt euros monsieur.", "fr"),
        (
            "Cela vous fera 20,15€ monsieur.",
            "Cela vous fera vingt euros et quinze centimes monsieur.",
            "fr",
        ),
        ("Ce sera 100.000,5.", "Ce sera cent mille virgule cinq.", "fr"),
        # German
        ("In 12,5 Sekunden.", "In zwölf Komma fünf Sekunden.", "de"),
        ("Es gab 50 Soldaten.", "Es gab fünfzig Soldaten.", "de"),
        ("Dies ist ein 1. Test", "Dies ist ein erste Test", "de"),  # Issue with gender
        ("Das macht $20 Herr.", "Das macht zwanzig Dollar Herr.", "de"),
        ("Das macht 20€ Herr.", "Das macht zwanzig Euro Herr.", "de"),
        (
            "Das macht 20,15€ Herr.",
            "Das macht zwanzig Euro und fünfzehn Cent Herr.",
            "de",
        ),
        # Spanish
        ("En 12,5 segundos.", "En doce punto cinco segundos.", "es"),
        ("Había 50 soldados.", "Había cincuenta soldados.", "es"),
        ("Este es un 1er test", "Este es un primero test", "es"),
        ("Eso le costará $20 señor.", "Eso le costará veinte dólares señor.", "es"),
        ("Eso le costará 20€ señor.", "Eso le costará veinte euros señor.", "es"),
        (
            "Eso le costará 20,15€ señor.",
            "Eso le costará veinte euros con quince céntimos señor.",
            "es",
        ),
        # Italian
        ("In 12,5 secondi.", "In dodici virgola cinque secondi.", "it"),
        ("C'erano 50 soldati.", "C'erano cinquanta soldati.", "it"),
        ("Questo è un 1° test", "Questo è un primo test", "it"),
        ("Ti costerà $20 signore.", "Ti costerà venti dollari signore.", "it"),
        ("Ti costerà 20€ signore.", "Ti costerà venti euro signore.", "it"),
        (
            "Ti costerà 20,15€ signore.",
            "Ti costerà venti euro e quindici centesimi signore.",
            "it",
        ),
        # Portuguese
        ("Em 12,5 segundos.", "Em doze vírgula cinco segundos.", "pt"),
        ("Havia 50 soldados.", "Havia cinquenta soldados.", "pt"),
        ("Este é um 1º teste", "Este é um primeiro teste", "pt"),
        ("Isso custará $20 senhor.", "Isso custará vinte dólares senhor.", "pt"),
        ("Isso custará 20€ senhor.", "Isso custará vinte euros senhor.", "pt"),
        (
            "Isso custará 20,15€ senhor.",
            "Isso custará vinte euros e quinze cêntimos senhor.",
            "pt",
        ),  # "cêntimos" should be "centavos" num2words issue
        # Polish
        ("W 12,5 sekundy.", "W dwanaście przecinek pięć sekundy.", "pl"),
        ("Było 50 żołnierzy.", "Było pięćdziesiąt żołnierzy.", "pl"),
        (
            "To będzie kosztować 20€ panie.",
            "To będzie kosztować dwadzieścia euro panie.",
            "pl",
        ),
        (
            "To będzie kosztować 20,15€ panie.",
            "To będzie kosztować dwadzieścia euro, piętnaście centów panie.",
            "pl",
        ),
        # Arabic
        ("في الـ 12,5 ثانية.", "في الـ اثنا عشر  , خمسون ثانية.", "ar"),
        ("كان هناك 50 جنديًا.", "كان هناك خمسون جنديًا.", "ar"),
        # ("ستكون النتيجة $20 يا سيد.", 'ستكون النتيجة عشرون دولار يا سيد.', 'ar'), # $ and € are mising from num2words
        # ("ستكون النتيجة 20€ يا سيد.", 'ستكون النتيجة عشرون يورو يا سيد.', 'ar'),
        # Czech
        ("Za 12,5 vteřiny.", "Za dvanáct celá pět vteřiny.", "cs"),
        ("Bylo tam 50 vojáků.", "Bylo tam padesát vojáků.", "cs"),
        ("To bude stát 20€ pane.", "To bude stát dvacet euro pane.", "cs"),
        ("To bude 20.15€ pane.", "To bude dvacet euro, patnáct centů pane.", "cs"),
        # Russian
        ("Через 12.5 секунды.", "Через двенадцать запятая пять секунды.", "ru"),
        ("Там было 50 солдат.", "Там было пятьдесят солдат.", "ru"),
        (
            "Это будет 20.15€ сэр.",
            "Это будет двадцать евро, пятнадцать центов сэр.",
            "ru",
        ),
        (
            "Это будет стоить 20€ господин.",
            "Это будет стоить двадцать евро господин.",
            "ru",
        ),
        # Dutch
        ("In 12,5 seconden.", "In twaalf komma vijf seconden.", "nl"),
        ("Er waren 50 soldaten.", "Er waren vijftig soldaten.", "nl"),
        ("Dat wordt dan $20 meneer.", "Dat wordt dan twintig dollar meneer.", "nl"),
        ("Dat wordt dan 20€ meneer.", "Dat wordt dan twintig euro meneer.", "nl"),
        # Chinese (Simplified)
        ("在12.5秒内", "在十二点五秒内", "zh"),
        ("有50名士兵", "有五十名士兵", "zh"),
        # ("那将是$20先生", '那将是二十美元先生', 'zh'), currency doesn't work
        # ("那将是20€先生", '那将是二十欧元先生', 'zh'),
        # Turkish
        # ("12,5 saniye içinde.", 'On iki virgül beş saniye içinde.', 'tr'), # decimal doesn't work for TR
        ("50 asker vardı.", "elli asker vardı.", "tr"),
        ("Bu 1. test", "Bu birinci test", "tr"),
        # ("Bu 100.000,5.", 'Bu yüz bin virgül beş.', 'tr'),
        # Hungarian
        ("12,5 másodperc alatt.", "tizenkettő egész öt tized másodperc alatt.", "hu"),
        ("50 katona volt.", "ötven katona volt.", "hu"),
        ("Ez az 1. teszt", "Ez az első teszt", "hu"),
        # Korean
        ("12.5 초 안에.", "십이 점 다섯 초 안에.", "ko"),
        ("50 명의 병사가 있었다.", "오십 명의 병사가 있었다.", "ko"),
        ("이것은 1 번째 테스트입니다", "이것은 첫 번째 테스트입니다", "ko"),
    ]
    for a, b, lang in test_cases:
        out = expand_numbers_multilingual(a, lang=lang)
        assert out == b, f"'{out}' vs '{b}'"

def test_abbreviations_multilingual():
    test_cases = [
        # English
        ("Hello Mr. Smith.", "Hello mister Smith.", "en"),
        ("Dr. Jones is here.", "doctor Jones is here.", "en"),
        # Spanish
        ("Hola Sr. Garcia.", "Hola señor Garcia.", "es"),
        ("La Dra. Martinez es muy buena.", "La doctora Martinez es muy buena.", "es"),
        # French
        ("Bonjour Mr. Dupond.", "Bonjour monsieur Dupond.", "fr"),
        (
            "Mme. Moreau est absente aujourd'hui.",
            "madame Moreau est absente aujourd'hui.",
            "fr",
        ),
        # German
        ("Frau Dr. Müller ist sehr klug.", "Frau doktor Müller ist sehr klug.", "de"),
        # Portuguese
        ("Olá Sr. Silva.", "Olá senhor Silva.", "pt"),
        (
            "Dra. Costa, você está disponível?",
            "doutora Costa, você está disponível?",
            "pt",
        ),
        # Italian
        ("Buongiorno, Sig. Rossi.", "Buongiorno, signore Rossi.", "it"),
        # ("Sig.ra Bianchi, posso aiutarti?", 'signora Bianchi, posso aiutarti?', 'it'), # Issue with matching that pattern
        # Polish
        ("Dzień dobry, P. Kowalski.", "Dzień dobry, pani Kowalski.", "pl"),
        (
            "M. Nowak, czy mogę zadać pytanie?",
            "pan Nowak, czy mogę zadać pytanie?",
            "pl",
        ),
        # Czech
        ("P. Novák", "pan Novák", "cs"),
        ("Dr. Vojtěch", "doktor Vojtěch", "cs"),
        # Dutch
        ("Dhr. Jansen", "de heer Jansen", "nl"),
        ("Mevr. de Vries", "mevrouw de Vries", "nl"),
        # Russian
        ("Здравствуйте Г-н Иванов.", "Здравствуйте господин Иванов.", "ru"),
        (
            "Д-р Смирнов здесь, чтобы увидеть вас.",
            "доктор Смирнов здесь, чтобы увидеть вас.",
            "ru",
        ),
        # Turkish
        ("Merhaba B. Yılmaz.", "Merhaba bay Yılmaz.", "tr"),
        ("Dr. Ayşe burada.", "doktor Ayşe burada.", "tr"),
        # Hungarian
        ("Dr. Szabó itt van.", "doktor Szabó itt van.", "hu"),
    ]

    for a, b, lang in test_cases:
        out = expand_abbreviations_multilingual(a, lang=lang)
        assert out == b, f"'{out}' vs '{b}'"


def test_symbols_multilingual():
    test_cases = [
        ("I have 14% battery", "I have 14 percent battery", "en"),
        ("Te veo @ la fiesta", "Te veo arroba la fiesta", "es"),
        ("J'ai 14° de fièvre", "J'ai 14 degrés de fièvre", "fr"),
        ("Die Rechnung beträgt £ 20", "Die Rechnung beträgt pfund 20", "de"),
        (
            "O meu email é ana&joao@gmail.com",
            "O meu email é ana e joao arroba gmail.com",
            "pt",
        ),
        (
            "linguaggio di programmazione C#",
            "linguaggio di programmazione C cancelletto",
            "it",
        ),
        ("Moja temperatura to 36.6°", "Moja temperatura to 36.6 stopnie", "pl"),
        ("Mám 14% baterie", "Mám 14 procento baterie", "cs"),
        ("Těším se na tebe @ party", "Těším se na tebe na party", "cs"),
        ("У меня 14% заряда", "У меня 14 процентов заряда", "ru"),
        ("Я буду @ дома", "Я буду собака дома", "ru"),
        ("Ik heb 14% batterij", "Ik heb 14 procent batterij", "nl"),
        ("Ik zie je @ het feest", "Ik zie je bij het feest", "nl"),
        ("لدي 14% في البطارية", "لدي 14 في المئة في البطارية", "ar"),
        ("我的电量为 14%", "我的电量为 14 百分之", "zh"),
        ("Pilim %14 dolu.", "Pilim yüzde 14 dolu.", "tr"),
        (
            "Az akkumulátorom töltöttsége 14%",
            "Az akkumulátorom töltöttsége 14 százalék",
            "hu",
        ),
        ("배터리 잔량이 14%입니다.", "배터리 잔량이 14 퍼센트입니다.", "ko"),
    ]

    for a, b, lang in test_cases:
        out = expand_symbols_multilingual(a, lang=lang)
        assert out == b, f"'{out}' vs '{b}'"

if __name__ == "__main__":
    test_expand_numbers_multilingual()
    test_abbreviations_multilingual()
    test_symbols_multilingual()
