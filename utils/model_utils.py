from transformers import AutoConfig, AutoModel

def encoder_model_setting(model_name, isPreTrain):
    model_config = AutoConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    encoder = basemodel.encoder

    return encoder, model_config

def decoder_model_setting(model_name, isPreTrain):
    model_config = AutoConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    decoder = basemodel.decoder

    return decoder, model_config

def return_model_name(model_type):
    if model_type == 'bert':
        out = 'bert-base-cased'
    if model_type == 'albert':
        out = 'textattack/albert-base-v2-imdb'
    if model_type == 'deberta':
        out = 'microsoft/deberta-v3-base'
    if model_type == 'bart':
        out = 'facebook/bart-large'
    if model_type == 'kr_bart':
        out = 'cosmoquester/bart-ko-mini'
    if model_type == 'T5':
        out = 't5-base'
    return out