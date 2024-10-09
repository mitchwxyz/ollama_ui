import re

quant_re = re.compile('[-_](Q3_K_(L|XL)|IQ4_XS|Q4_(K_(S|M|L)|0)|Q5_K_(S|M|L)|Q6_K(_L)?|Q8_0|F16)')

def get_defaults(model: str)-> dict:
    if match := quant_re.search(model):
        quant_location = match.span()[0]
        base_model = model[:quant_location]
    else:
        base_model = model
    
    if base_model == "Llama-3.1-8B-Instruct":
        return {
            "ICON": "🦙",
            "TEMPERATURE": 0.6,
            "TOP_K": 50,
            "TOP_P": 0.9,
            "MIN_P": 0.02,
            "TYPICAL_P": 0.5,
            "NUM_CTX": 1024*8,
            "NUM_PREDICT": 256,
            "REPEAT_LAST_N": 128,
            "REPEAT_PENALTY": 1.02,
            "MIROSTAT": 0,
            "MIROSTAT_ETA": 0.0,
            "MIROSTAT_TAU": 0.0,
        }
    elif base_model == "DeepSeek-Coder-V2-Lite-Instruct":
        return {
            "ICON": "🧮",
            "TEMPERATURE": 0.4,
            "TOP_K": 60,
            "TOP_P": 0.9,
            "MIN_P": 0.0,
            "TYPICAL_P": 0.5,
            "NUM_CTX": 1024*8,
            "NUM_PREDICT": 2048,
            "REPEAT_LAST_N": 8,
            "REPEAT_PENALTY": 1.0,
            "MIROSTAT": 0,
            "MIROSTAT_ETA": 0.0,
            "MIROSTAT_TAU": 0.0,
        }
    else:
        return {
            "ICON": "🤖",
            "TEMPERATURE": 0.7,
            "TOP_K": 40,
            "TOP_P": 0.9,
            "MIN_P": 0.02,
            "TYPICAL_P": 0.75,
            "NUM_CTX": 1024*8,
            "NUM_PREDICT": 128,
            "REPEAT_LAST_N": 64,
            "REPEAT_PENALTY": 1.21,
            "MIROSTAT": 1,
            "MIROSTAT_ETA": 0.10,
            "MIROSTAT_TAU": 4.0,
        }