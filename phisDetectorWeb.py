import datetime
import re
from urllib.parse import urlparse
import httpx
import whois
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def havingIP(url):
    ip_pattern = r"(?:http[s]?://)?(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
    return 1 if re.match(ip_pattern, url) else 0

def haveAtSign(url):
    return 1 if '@' in url else 0

def uses_https(url):
    return 1 if urlparse(url).scheme == "https" else 0

def is_suspicious_tld(url):
    suspicious_tlds = {
        "tk", "ml", "ga", "cf", "gq",
        "xyz", "top", "club", "click", "work",
        "support", "fit", "loan", "download", "men",
        "review", "date", "party", "trade", "stream",
        "gdn", "win", "accountant", "science", "racing",
        "buzz", "icu", "wang", "live", "host", "info", "lat"
    }
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        domain_parts = netloc.lower().split('.')
        tld = domain_parts[-1] if domain_parts else ""
        return 1 if tld in suspicious_tlds else 0
    except:
        return 0

def getLength(url):
    return len(url)

def getDepth(url):
    return len([i for i in urlparse(url).path.split('/') if i])

def tinyURL(url):
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                          r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                          r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                          r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.ly|cutt\.ly|" \
                          r"u\.to|v\.gd|qr\.ae|adf\.ly|bitly\.com|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|" \
                          r"u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|" \
                          r"1url\.com|tweez\.me|v\.gd|link\.zip\.net"
    return 1 if re.search(shortening_services, url) else 0

def prefixSuffix(url):
    return 1 if '-' in urlparse(url).netloc else 0

def no_of_dots(url):
    return urlparse(url).netloc.count('.')

def has_unicode(url):
    return 1 if urlparse(url).netloc.startswith("xn--") else 0

def count_numbers(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    return sum(c.isdigit() for c in domain)

def domainAge(domain_name):
    try:
        w = whois.whois(domain_name)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        return (datetime.datetime.now() - creation_date).days
    except:
        return 250

def domainEnd(domain_name):
    try:
        w = whois.whois(domain_name)
        expiration_date = w.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        return (expiration_date - datetime.datetime.now()).days
    except:
        return 250

def redirect_count(url):
    try:
        r = httpx.get(url, follow_redirects=True, timeout=5)
        return len(r.history)
    except:
        return 1

def final_url_differs(url):
    try:
        r = httpx.get(url, follow_redirects=True, timeout=5)
        return 1 if str(r.url) != url else 0
    except:
        return 1
    
def featureExtraction(url):
    try:
        domain = urlparse(url).netloc
        feature_dict = {
            'Uses_HTTPS': uses_https(url),
            'URL_tld': is_suspicious_tld(url),
            #'URL_Length': getLength(url),
            'URL_Depth': getDepth(url),
            'TinyURL': tinyURL(url),
            'Prefix/Suffix': prefixSuffix(url),
            'No_Of_Dots': no_of_dots(url),
            'Domain_Age': domainAge(domain),
            'Domain_End': domainEnd(domain),
            'Have_Symbol': has_unicode(url) + haveAtSign(url) + havingIP(url),
            'Redirect_Count': redirect_count(url),
            'Final_URL_Differs': final_url_differs(url),
            'Num_Count': count_numbers(url)
        }
        return pd.DataFrame([feature_dict])
    except Exception as e:
        print("Error extracting features:", e)
        return pd.DataFrame()

# --- [Fuzzy Inference System] ---
def fuzzy_score(row, url=''):
    # Define fuzzy variables
    # url_length = ctrl.Antecedent(np.arange(0, 101, 1), 'url_length')
    url_depth = ctrl.Antecedent(np.arange(0, 10, 1), 'url_depth')
    dots = ctrl.Antecedent(np.arange(0, 10, 1), 'dots')
    symbols = ctrl.Antecedent(np.arange(0, 4, 1), 'symbols')
    redirects = ctrl.Antecedent(np.arange(0, 5, 1), 'redirects')
    domain_age = ctrl.Antecedent(np.arange(0, 1001, 1), 'domain_age')
    domain_end = ctrl.Antecedent(np.arange(0, 366, 1), 'domain_end')
    num_count = ctrl.Antecedent(np.arange(0, 15, 1), 'num_count')
    https = ctrl.Antecedent(np.arange(0, 2, 1), 'https')
    tinyurl = ctrl.Antecedent(np.arange(0, 2, 1), 'tinyurl')
    prefix_suffix = ctrl.Antecedent(np.arange(0, 2, 1), 'prefix_suffix')
    tld_suspicious = ctrl.Antecedent(np.arange(0, 2, 1), 'tld_suspicious')
    final_url_diff = ctrl.Antecedent(np.arange(0, 2, 1), 'final_url_diff')

    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # Membership Functions
    # url_length['short'] = fuzz.trimf(url_length.universe, [0, 0, 50])
    # url_length['medium'] = fuzz.trimf(url_length.universe, [30, 60, 90])
    # url_length['long'] = fuzz.trimf(url_length.universe, [70, 100, 100])

    url_depth['shallow'] = fuzz.trimf(url_depth.universe, [0, 0, 3])
    url_depth['medium'] = fuzz.trimf(url_depth.universe, [2, 4, 6])
    url_depth['deep'] = fuzz.trimf(url_depth.universe, [5, 9, 9])

    dots['few'] = fuzz.trimf(dots.universe, [0, 0, 3])
    dots['many'] = fuzz.trimf(dots.universe, [2, 5, 9])

    symbols['none'] = fuzz.trimf(symbols.universe, [0, 0, 1])
    symbols['some'] = fuzz.trimf(symbols.universe, [1, 2, 3])
    symbols['many'] = fuzz.trimf(symbols.universe, [2, 3, 3])

    redirects['low'] = fuzz.trimf(redirects.universe, [0, 0, 2])
    redirects['high'] = fuzz.trimf(redirects.universe, [2, 4, 4])

    domain_age['young'] = fuzz.trimf(domain_age.universe, [0, 0, 180])
    domain_age['old'] = fuzz.trimf(domain_age.universe, [180, 1000, 1000])

    domain_end['soon'] = fuzz.trimf(domain_end.universe, [0, 0, 30])
    domain_end['far'] = fuzz.trimf(domain_end.universe, [30, 365, 365])

    num_count['low'] = fuzz.trimf(num_count.universe, [0, 0, 3])
    num_count['high'] = fuzz.trimf(num_count.universe, [3, 10, 14])

    https['no'] = fuzz.trimf(https.universe, [0, 0, 0])
    https['yes'] = fuzz.trimf(https.universe, [1, 1, 1])

    tinyurl['no'] = fuzz.trimf(tinyurl.universe, [0, 0, 0])
    tinyurl['yes'] = fuzz.trimf(tinyurl.universe, [1, 1, 1])

    prefix_suffix['no'] = fuzz.trimf(prefix_suffix.universe, [0, 0, 0])
    prefix_suffix['yes'] = fuzz.trimf(prefix_suffix.universe, [1, 1, 1])

    tld_suspicious['no'] = fuzz.trimf(tld_suspicious.universe, [0, 0, 0])
    tld_suspicious['yes'] = fuzz.trimf(tld_suspicious.universe, [1, 1, 1])

    final_url_diff['no'] = fuzz.trimf(final_url_diff.universe, [0, 0, 0])
    final_url_diff['yes'] = fuzz.trimf(final_url_diff.universe, [1, 1, 1])

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 30])
    risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 70])
    risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

    # Fuzzy rules (expand as needed)
    rules = [
        ctrl.Rule(dots['many'] | num_count['high'], risk['high']),
        ctrl.Rule(symbols['many'] | redirects['high'], risk['high']),
        ctrl.Rule(domain_age['young'] | domain_end['soon'], risk['high']),
        ctrl.Rule(url_depth['deep'] & dots['many'], risk['medium']),
        ctrl.Rule(symbols['none'] & redirects['low'] & domain_age['old'], risk['low']),
        ctrl.Rule(https['no'], risk['medium']),
        ctrl.Rule(tinyurl['yes'] | prefix_suffix['yes'], risk['medium']),
        ctrl.Rule(tld_suspicious['yes'], risk['high']),
        ctrl.Rule(final_url_diff['yes'], risk['medium']),
    ]

    control_system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(control_system)

    # Masukkan input dari fitur hasil ekstraksi
    # sim.input['url_length'] = row['URL_Length']
    sim.input['url_depth'] = row['URL_Depth']
    sim.input['dots'] = row['No_Of_Dots']
    sim.input['symbols'] = row['Have_Symbol']
    sim.input['redirects'] = row['Redirect_Count']
    sim.input['domain_age'] = row['Domain_Age']
    sim.input['domain_end'] = row['Domain_End']
    sim.input['num_count'] = row['Num_Count']
    sim.input['https'] = row['Uses_HTTPS']
    sim.input['tinyurl'] = row['TinyURL']
    sim.input['prefix_suffix'] = row['Prefix/Suffix']
    sim.input['tld_suspicious'] = row['URL_tld']
    sim.input['final_url_diff'] = row['Final_URL_Differs']

    sim.compute()
    fuzzy_val = sim.output['risk']
    score = round(fuzzy_val / 100.0, 3)

    # Breakdown dictionary
    breakdown = {
        #'URL_Length': row['URL_Length'],
        'URL_Depth': row['URL_Depth'],
        'No_Of_Dots': row['No_Of_Dots'],
        'Have_Symbol': row['Have_Symbol'],
        'Redirect_Count': row['Redirect_Count'],
        'Domain_Age': row['Domain_Age'],
        'Domain_End': row['Domain_End'],
        'Num_Count': row['Num_Count'],
        'Uses_HTTPS': row['Uses_HTTPS'],
        'TinyURL': row['TinyURL'],
        'Prefix/Suffix': row['Prefix/Suffix'],
        'URL_tld': row['URL_tld'],
        'Final_URL_Differs': row['Final_URL_Differs']
    }

    return score, breakdown

# --- [Rescaling to percentage for UI] ---
def compute_percentage(score):
    return round(score * 100, 1)